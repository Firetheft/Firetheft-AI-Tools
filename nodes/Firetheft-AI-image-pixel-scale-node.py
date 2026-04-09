import logging
import nodes
import torch
import comfy.utils
import math
import folder_paths
import os
import uuid
import comfy.model_management
from typing import TypedDict, Optional, List
from comfy_api.latest import io, Types

# --- Internal Helper Functions (Logic Kept Same) ---
def image_upscale_on_pixel_space(image, scale_method, scale_factor):
    w = max(2, int(round((image.shape[2] * scale_factor) / 2.0)) * 2)
    h = max(2, int(round((image.shape[1] * scale_factor) / 2.0)) * 2)
    image = nodes.ImageScale().upscale(image, scale_method, w, h, False)[0]
    return image

def image_upscale_on_pixel_space_with_model_optimized(image, scale_method, upscale_model, scale_factor, max_megapixels, chunk_frames, cache_backend):
    batch_size = image.shape[0]
    w = image.shape[2]
    h = image.shape[1]

    new_w = max(2, int(round((w * scale_factor) / 2.0)) * 2)
    new_h = max(2, int(round((h * scale_factor) / 2.0)) * 2)
    
    max_pixels = int(max_megapixels * 1024 * 1024)
    out_pixels_per_frame = new_w * new_h
    batch_size_per_step = max(1, max_pixels // out_pixels_per_frame)
    
    total_steps = math.ceil(batch_size / batch_size_per_step)
    pbar = comfy.utils.ProgressBar(total_steps)

    out_images = []
    temp_files = []
    
    temp_dir = folder_paths.get_temp_directory()
    processed_count = 0

    try:
        total_chunks = math.ceil(batch_size / chunk_frames)
        for chunk_idx in range(total_chunks):
            chunk_start = chunk_idx * chunk_frames
            chunk_end = min(chunk_start + chunk_frames, batch_size)
            curr_chunk = image[chunk_start:chunk_end]
            
            chunk_results = []
            
            for i in range(0, curr_chunk.shape[0], batch_size_per_step):
                batch = curr_chunk[i:i + batch_size_per_step]
                actual_batch_len = batch.shape[0]
                curr_w = w
                
                batch = batch.to(comfy.model_management.get_torch_device())
                
                while curr_w < new_w:
                    model_upscaler = nodes.NODE_CLASS_MAPPINGS['ImageUpscaleWithModel']()
                    if hasattr(model_upscaler, 'execute'):
                        batch = model_upscaler.execute(upscale_model, batch)[0]
                    else:
                        batch = model_upscaler.upscale(upscale_model, batch)[0]

                    if batch.shape[2] == curr_w:
                        break
                    curr_w = batch.shape[2]

                batch = nodes.ImageScale().upscale(batch, scale_method, new_w, new_h, False)[0]
                chunk_results.append(batch.cpu())
                del batch
                
                processed_count += actual_batch_len
                pbar.update(1)
                comfy.model_management.soft_empty_cache()
            
            processed_chunk = torch.cat(chunk_results, dim=0)
            
            if cache_backend == "disk":
                file_path = os.path.join(temp_dir, f"firetheft_upscale_chunk_{chunk_idx}_{uuid.uuid4()}.pt")
                torch.save(processed_chunk, file_path)
                temp_files.append(file_path)
                del processed_chunk
            else:
                out_images.append(processed_chunk)
                del processed_chunk
            
        if cache_backend == "disk":
            final_images = []
            for f in temp_files:
                final_images.append(torch.load(f))
            image = torch.cat(final_images, dim=0)
        else:
            image = torch.cat(out_images, dim=0)
            
    finally:
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass

    return image

# --- New API Node Definition ---

class ScaleModeInput(TypedDict, total=False):
    scale_mode: str
    scale_factor: float
    resolution: str

class ImagePixelScaleNode(io.ComfyNode):
    upscale_methods = ["lanczos", "bicubic", "bilinear", "nearest-exact", "area"]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImagePixelScaleNode",
            display_name="Image Pixel Scale（图像像素空间缩放）",
            category="📜Firetheft AI Tools",
            inputs=[
                io.Image.Input("image"),
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
                io.Float.Input("max_megapixels", default=32.0, min=0.01, max=1024.0, step=0.01, tooltip="Maximum pixel budget (MP)."),
                io.Int.Input("chunk_frames", default=128, min=1, max=65535, step=1, tooltip="Outer chunk frame count."),
                io.Combo.Input("cache_backend", options=["memory", "disk"], default="disk", tooltip="Cache mode: memory or disk."),
                io.UpscaleModel.Input("upscale_model_opt", optional=True, tooltip="(Optional) External upscale model.")
            ],
            outputs=[
                io.Image.Output("images")
            ]
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        scale_mode: ScaleModeInput,
        scale_method: str,
        max_megapixels: float,
        chunk_frames: int,
        cache_backend: str,
        upscale_model_opt: Optional[torch.Tensor] = None
    ) -> io.NodeOutput:
        
        mode = scale_mode["scale_mode"]
        
        if mode == "resolution":
            try:
                resolution_str = scale_mode.get("resolution", "2k (2560)")
                target_longest_side = int(resolution_str.split("(")[1].replace(")", ""))
                longest_side = max(image.shape[2], image.shape[1])
                actual_scale_factor = target_longest_side / longest_side
            except Exception:
                actual_scale_factor = 1.0
        else:
            actual_scale_factor = scale_mode.get("scale_factor", 2.0)

        if upscale_model_opt is None:
            image = image_upscale_on_pixel_space(image, scale_method, actual_scale_factor)
        else:
            image = image_upscale_on_pixel_space_with_model_optimized(
                image, scale_method, upscale_model_opt, actual_scale_factor, 
                max_megapixels, chunk_frames, cache_backend
            )
        
        return io.NodeOutput(image)

# Compatibility Mappings for older loaders if needed (optional since io.ComfyNode exports automatically)
NODE_CLASS_MAPPINGS = {
    "ImagePixelScaleNode": ImagePixelScaleNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePixelScaleNode": "Image Pixel Scale（图像像素空间缩放）"
}
