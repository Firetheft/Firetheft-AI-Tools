import logging
import nodes
import torch
import comfy.utils
import math
import folder_paths
import os
import uuid
import comfy.model_management

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
    
    # Human-readable MP to raw pixels
    max_pixels = int(max_megapixels * 1024 * 1024)
    
    # Inner batching logic based on pixels
    out_pixels_per_frame = new_w * new_h
    batch_size_per_step = max(1, max_pixels // out_pixels_per_frame)
    
    # Progress is now based on total steps, not just chunks
    total_steps = math.ceil(batch_size / batch_size_per_step)
    pbar = comfy.utils.ProgressBar(total_steps)

    out_images = []
    temp_files = []
    
    temp_dir = folder_paths.get_temp_directory()
    processed_count = 0

    try:
        # Outer chunking loop (Grouping for cache/memory management)
        total_chunks = math.ceil(batch_size / chunk_frames)
        for chunk_idx in range(total_chunks):
            chunk_start = chunk_idx * chunk_frames
            chunk_end = min(chunk_start + chunk_frames, batch_size)
            curr_chunk = image[chunk_start:chunk_end]
            
            chunk_results = []
            
            # Inner processing loop (Model batching)
            for i in range(0, curr_chunk.shape[0], batch_size_per_step):
                batch = curr_chunk[i:i + batch_size_per_step]
                actual_batch_len = batch.shape[0]
                curr_w = w
                
                # Move to device ONLY for processing
                batch = batch.to(comfy.model_management.get_torch_device())
                
                while curr_w < new_w:
                    model_upscaler = nodes.NODE_CLASS_MAPPINGS['ImageUpscaleWithModel']()
                    if hasattr(model_upscaler, 'execute'):
                        batch = model_upscaler.execute(upscale_model, batch)[0]
                    else:
                        batch = model_upscaler.upscale(upscale_model, batch)[0]

                    if batch.shape[2] == curr_w:
                        logging.info("[image_upscale_on_pixel_space_with_model] x1 upscale model selected or no growth")
                        break
                    curr_w = batch.shape[2]

                # Final resize for this batch
                batch = nodes.ImageScale().upscale(batch, scale_method, new_w, new_h, False)[0]
                
                # Collect batch result on CPU
                chunk_results.append(batch.cpu())
                del batch
                
                # Update progress per batch
                processed_count += actual_batch_len
                pbar.update(1)
                logging.info(f"Upscaling: {processed_count}/{batch_size} frames processed (Current batch size: {actual_batch_len})")
                
                # Proactive VRAM clearing after each batch step
                comfy.model_management.soft_empty_cache()
            
            # Processed a full chunk, combine and cache
            processed_chunk = torch.cat(chunk_results, dim=0)
            
            if cache_backend == "disk":
                file_path = os.path.join(temp_dir, f"firetheft_upscale_chunk_{chunk_idx}_{uuid.uuid4()}.pt")
                torch.save(processed_chunk, file_path)
                temp_files.append(file_path)
                del processed_chunk
            else:
                out_images.append(processed_chunk)
                del processed_chunk
            
            logging.info(f"Chunk Cache: Group {chunk_idx + 1}/{total_chunks} saved to {cache_backend}")

        # Final reconstruction (Cat everything into one tensor)
        if cache_backend == "disk":
            final_images = []
            for f in temp_files:
                final_images.append(torch.load(f))
            image = torch.cat(final_images, dim=0)
        else:
            image = torch.cat(out_images, dim=0)
            
    finally:
        # Cleanup temp files
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass

    return image

class ImagePixelScaleNode:
    upscale_methods = ["lanczos", "bicubic", "bilinear", "nearest-exact", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "image": ("IMAGE", ),
                     "scale_mode": (["multiple", "resolution"], {"default": "multiple", "tooltip": "Scale mode: by multiple, or by fixed resolution long side automatically adapts."}),
                     "scale_factor": ("FLOAT", {"default": 2, "min": 0.1, "max": 10000, "step": 0.05, "tooltip": "Target scale factor (e.g., 2 represents 2 times magnification)."}),
                     "resolution": (["720p (1280)", "1080p (1920)", "2k (2560)", "3k (3072)", "4k (3840)", "8k (7680)"], {"default": "2k (2560)", "tooltip": "Target fixed resolution (based on the long side in pixels, maintaining the original image ratio)."}),
                     "scale_method": (s.upscale_methods, {"default": "bicubic", "tooltip": "Scaling algorithm: for better image quality, it is recommended to use lanczos or bicubic to avoid the jaggedness of nearest-exact."}),
                     "max_megapixels": ("FLOAT", {"default": 32, "min": 0.01, "max": 1024.0, "step": 0.01, "tooltip": "Maximum pixel budget (MP). The system automatically calculates the batch based on this value. 8G VRAM is recommended for 16-32, and the larger the value, the faster the speed but the more VRAM it occupies."}),
                     "chunk_frames": ("INT", {"default": 128, "min": 1, "max": 65535, "step": 1, "tooltip": "Outer chunk processing frame count. Controls the frequency of progress updates and the size of disk cache units."}),
                     "cache_backend": (["memory", "disk"], {"default": "disk", "tooltip": "Cache mode: memory (fast, uses RAM) or disk (slower, uses disk, prevents OOM)."}),
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", {"tooltip": "(Optional) External upscale model. If connected, it will be used for the initial upscale before precise dimension adjustment."}),
                    }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doit"

    CATEGORY = "📜Firetheft AI Tools"

    def doit(self, image, scale_mode, scale_factor, resolution, scale_method, max_megapixels, chunk_frames, cache_backend, upscale_model_opt=None):
        if scale_mode == "resolution":
            try:
                target_longest_side = int(resolution.split("(")[1].replace(")", ""))
                longest_side = max(image.shape[2], image.shape[1])
                actual_scale_factor = target_longest_side / longest_side
            except Exception:
                actual_scale_factor = scale_factor
        else:
            actual_scale_factor = scale_factor

        if upscale_model_opt is None:
            image = image_upscale_on_pixel_space(image, scale_method, actual_scale_factor)
        else:
            image = image_upscale_on_pixel_space_with_model_optimized(
                image, scale_method, upscale_model_opt, actual_scale_factor, 
                max_megapixels, chunk_frames, cache_backend
            )
        return (image,)

NODE_CLASS_MAPPINGS = {
    "ImagePixelScaleNode": ImagePixelScaleNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePixelScaleNode": "Image Pixel Scale（图像像素空间缩放）"
}
