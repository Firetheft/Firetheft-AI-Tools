from comfy_extras.nodes_lt import get_noise_mask, LTXVAddGuide
import torch
import comfy.utils
from comfy_api.latest import io
from typing import TypedDict, List

class FiretheftLTXSequencer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        # Static inputs at the top
        inputs = [
            io.Conditioning.Input("positive", tooltip="Positive conditioning"),
            io.Conditioning.Input("negative", tooltip="Negative conditioning"),
            io.Vae.Input("vae", tooltip="Video VAE"),
            io.Latent.Input("latent", tooltip="Video latent"),
            io.Image.Input("multi_input", tooltip="Images from MultiImageLoader"),
            io.Int.Input("frame_rate", default=24, min=1, max=120, display_name="FPS"),
        ]
        
        # Flattened options to ensure link stability across refreshes
        settings_options = []
        for n in range(1, 51):
            # Frames Option for n images
            f_widgets = []
            for i in range(1, n + 1):
                f_widgets.append(io.Int.Input(
                    f"frame_{i}", default=0, min=-9999, max=9999, step=1, 
                    display_name=f"Img #{i} Frame"
                ))
                f_widgets.append(io.Float.Input(
                    f"strength_{i}", default=1.0, min=0.0, max=1.0, step=0.001, 
                    display_name=f"Img #{i} Strength"
                ))
            settings_options.append(io.DynamicCombo.Option(f"{n:02} | Frames", f_widgets))
            
            # Seconds Option for n images
            s_widgets = []
            for i in range(1, n + 1):
                s_widgets.append(io.Float.Input(
                    f"second_{i}", default=0.0, min=0.0, max=9999.0, step=0.1, 
                    display_name=f"Img #{i} Second"
                ))
                s_widgets.append(io.Float.Input(
                    f"strength_{i}", default=1.0, min=0.0, max=1.0, step=0.001, 
                    display_name=f"Img #{i} Strength"
                ))
            settings_options.append(io.DynamicCombo.Option(f"{n:02} | Seconds", s_widgets))

        inputs.append(io.DynamicCombo.Input(
            "sequencer_settings",
            options=settings_options,
            display_name="Settings (Count | Mode)",
            tooltip="Select image count and insertion mode. Unified for link stability."
        ))

        return io.Schema(
            node_id="FiretheftLTXSequencer",
            display_name="Firetheft LTX Sequencer",
            category="📜Firetheft AI Tools/LTX",
            description="Ultra-stable LTX Sequencer with defined parameter limits and high precision.",
            inputs=inputs,
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, latent, multi_input, frame_rate, sequencer_settings, **kwargs) -> io.NodeOutput:
        # Parse the unified settings string
        settings_str = sequencer_settings.get("sequencer_settings", "01 | Frames")
        try:
            parts = settings_str.split(" | ")
            num_images = int(parts[0])
            insert_mode = parts[1].lower()
        except:
            num_images = 1
            insert_mode = "frames"

        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"].clone()
        
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"].clone()
        else:
            batch, _, latent_frames, latent_height, latent_width = latent_image.shape
            noise_mask = torch.ones((batch, 1, latent_frames, 1, 1), dtype=torch.float32, device=latent_image.device)

        _, _, latent_length, latent_height, latent_width = latent_image.shape
        batch_size = multi_input.shape[0] if multi_input is not None else 0

        for i in range(1, num_images + 1):
            if i > batch_size:
                continue

            img = multi_input[i-1:i]
            if img is None:
                continue

            f_idx = None
            if insert_mode == "frames":
                f_idx = sequencer_settings.get(f"frame_{i}")
            elif insert_mode == "seconds":
                sec = sequencer_settings.get(f"second_{i}")
                if sec is not None:
                    f_idx = int(sec * frame_rate)

            if f_idx is None:
                continue
            
            strength = sequencer_settings.get(f"strength_{i}", 1.0)

            image_1, t = LTXVAddGuide.encode(vae, latent_width, latent_height, img, scale_factors)
            frame_idx, latent_idx = LTXVAddGuide.get_latent_index(positive, latent_length, len(image_1), f_idx, scale_factors)
            assert latent_idx + t.shape[2] <= latent_length, "Conditioning frames exceed latent length."

            positive, negative, latent_image, noise_mask = LTXVAddGuide.append_keyframe(
                positive, negative, frame_idx, latent_image, noise_mask, t, strength, scale_factors,
            )

        return io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask})

NODE_CLASS_MAPPINGS = {"FiretheftLTXSequencer": FiretheftLTXSequencer}
NODE_DISPLAY_NAME_MAPPINGS = {"FiretheftLTXSequencer": "Firetheft LTX Sequencer"}
