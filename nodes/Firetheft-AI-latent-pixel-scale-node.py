import comfy.model_management
import inspect
import logging
import torch
import torch.nn.functional as F
import nodes
from nodes import MAX_RESOLUTION
from typing import TypedDict, Optional, List, Tuple
from comfy_api.latest import io, Types

# --- Internal Helper Functions (Logic Kept Same) ---

def vae_decode(vae, samples, use_tile, hook, tile_size=512, overlap=64):
    if use_tile:
        decoder = nodes.VAEDecodeTiled()
        if 'overlap' in inspect.signature(decoder.decode).parameters:
            pixels = decoder.decode(vae, samples, tile_size, overlap=overlap)[0]
        else:
            logging.warning("[Impact Pack] Your ComfyUI is outdated.")
            pixels = decoder.decode(vae, samples, tile_size)[0]
    else:
        pixels = nodes.VAEDecode().decode(vae, samples)[0]

    if hook is not None:
        pixels = hook.post_decode(pixels)

    return pixels

def vae_encode(vae, pixels, use_tile, hook, tile_size=512, overlap=64):
    if use_tile:
        encoder = nodes.VAEEncodeTiled()
        if 'overlap' in inspect.signature(encoder.encode).parameters:
            samples = encoder.encode(vae, pixels, tile_size, overlap=overlap)[0]
        else:
            logging.warning("[Impact Pack] Your ComfyUI is outdated.")
            samples = encoder.encode(vae, pixels, tile_size)[0]
    else:
        samples = nodes.VAEEncode().encode(vae, pixels)[0]

    if hook is not None:
        samples = hook.post_encode(samples)

    return samples

def latent_upscale_on_pixel_space2(samples, scale_method, scale_factor, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    pixels = vae_decode(vae, samples, use_tile, hook, tile_size=tile_size, overlap=overlap)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    w = pixels.shape[2] * scale_factor
    h = pixels.shape[1] * scale_factor
    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(w), int(h), False)[0]

    old_pixels = pixels
    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook, tile_size=tile_size, overlap=overlap), old_pixels

def latent_upscale_on_pixel_space_with_model2(samples, scale_method, upscale_model, scale_factor, vae, use_tile=False,
    tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    pixels = vae_decode(vae, samples, use_tile, hook, tile_size=tile_size, overlap=overlap)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    w = pixels.shape[2]
    h = pixels.shape[1]

    new_w = w * scale_factor
    new_h = h * scale_factor

    current_w = w
    while current_w < new_w:
        model_upscaler = nodes.NODE_CLASS_MAPPINGS['ImageUpscaleWithModel']()
        if hasattr(model_upscaler, 'execute'):
            pixels = model_upscaler.execute(upscale_model, pixels)[0]
        else:
            pixels = model_upscaler.upscale(upscale_model, pixels)[0]

        current_w = pixels.shape[2]
        if current_w == w:
            logging.info("[latent_upscale_on_pixel_space_with_model] x1 upscale model selected")
            break

    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(new_w), int(new_h), False)[0]

    old_pixels = pixels
    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook, tile_size=tile_size, overlap=overlap), old_pixels

# --- New API Node Definition ---

class ScaleModeInput(TypedDict, total=False):
    scale_mode: str
    scale_factor: float
    resolution: str

class LatentPixelScaleNode(io.ComfyNode):
    upscale_methods = ["lanczos", "bicubic", "bilinear", "nearest-exact", "area"]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentPixelScaleNode",
            display_name="Latent Pixel Scale（像素空间缩放）",
            category="📜Firetheft AI Tools",
            inputs=[
                io.Latent.Input("samples"),
                io.DynamicCombo.Input(
                    "scale_mode",
                    tooltip="Scale mode: by multiple, or by fixed resolution long side automatically adapts.",
                    options=[
                        io.DynamicCombo.Option("multiple", [
                            io.Float.Input("scale_factor", default=2.0, min=0.1, max=10000.0, step=0.05, tooltip="Target scale factor.")
                        ]),
                        io.DynamicCombo.Option("resolution", [
                            io.Combo.Input("resolution", options=["720p (1280)", "1080p (1920)", "2k (2560)", "3k (3072)", "4k (3840)", "8k (7680)"], default="2k (2560)", tooltip="Target fixed resolution.")
                        ])
                    ],
                ),
                io.Combo.Input("scale_method", options=cls.upscale_methods, default="bicubic", tooltip="Scaling algorithm."),
                io.Vae.Input("vae"),
                io.Boolean.Input("use_tiled_vae", default=False, label_on="enabled", label_off="disabled", tooltip="Use Tiled VAE for large images."),
                io.UpscaleModel.Input("upscale_model_opt", optional=True, tooltip="(Optional) External upscale model.")
            ],
            outputs=[
                io.Latent.Output("latent"),
                io.Image.Output("image")
            ]
        )

    @classmethod
    def execute(
        cls,
        samples: dict,
        scale_mode: ScaleModeInput,
        scale_method: str,
        vae: torch.Tensor,
        use_tiled_vae: bool,
        upscale_model_opt: Optional[torch.Tensor] = None
    ) -> io.NodeOutput:
        
        mode = scale_mode["scale_mode"]
        
        # Calculate actual scale factor
        if mode == "resolution":
            try:
                resolution_str = scale_mode.get("resolution", "2k (2560)")
                target_longest_side = int(resolution_str.split("(")[1].replace(")", ""))
                
                # Latent samples are usually 1/8 of pixel size
                # Shape is [B, C, H, W] where H,W are latent dimensions
                latent_h = samples['samples'].shape[2]
                latent_w = samples['samples'].shape[3]
                pixel_h = latent_h * 8
                pixel_w = latent_w * 8
                
                longest_side = max(pixel_h, pixel_w)
                actual_scale_factor = target_longest_side / longest_side
            except Exception:
                actual_scale_factor = 1.0
        else:
            actual_scale_factor = scale_mode.get("scale_factor", 2.0)

        # Process
        if upscale_model_opt is None:
            latent, image = latent_upscale_on_pixel_space2(
                samples, scale_method, actual_scale_factor, vae, use_tile=use_tiled_vae
            )
        else:
            latent, image = latent_upscale_on_pixel_space_with_model2(
                samples, scale_method, upscale_model_opt, actual_scale_factor, vae, use_tile=use_tiled_vae
            )
            
        return io.NodeOutput(latent, image)

# Compatibility Mappings
NODE_CLASS_MAPPINGS = {
    "LatentPixelScaleNode": LatentPixelScaleNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentPixelScaleNode": "Latent Pixel Scale（像素空间缩放）"
}