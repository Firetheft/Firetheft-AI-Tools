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
    w = image.shape[2] * scale_factor
    h = image.shape[1] * scale_factor
    image = nodes.ImageScale().upscale(image, scale_method, int(w), int(h), False)[0]
    return image

def image_upscale_on_pixel_space_with_model_optimized(image, scale_method, upscale_model, scale_factor, max_megapixels, chunk_frames, cache_backend):
    batch_size = image.shape[0]
    w = image.shape[2]
    h = image.shape[1]

    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
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
                     "scale_method": (s.upscale_methods, {"default": "bicubic", "tooltip": "缩放算法：注重画质建议使用 lanczos 或 bicubic，避免 nearest-exact 的锯齿感。"}),
                     "scale_factor": ("FLOAT", {"default": 2, "min": 0.1, "max": 10000, "step": 0.05, "tooltip": "目标缩放倍数（例如 2 代表放大 2 倍）。"}),
                     "max_megapixels": ("FLOAT", {"default": 32, "min": 0.01, "max": 1024.0, "step": 0.01, "tooltip": "最大像素预算 (MP)。系统根据此值自动计算批次。8G 显存建议 16-32，数值越大速度越快但越占显存。"}),
                     "chunk_frames": ("INT", {"default": 128, "min": 1, "max": 65535, "step": 1, "tooltip": "外层分块处理帧数。控制进度更新频率和磁盘缓存的单元大小。"}),
                     "cache_backend": (["memory", "disk"], {"default": "disk", "tooltip": "缓存模式：memory（放内存，快）或 disk（暂存硬盘，防显存/内存溢出）。"}),
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", {"tooltip": "（可选）外接放大模型。如果连接，将先使用模型放大，再精确调整到目标尺寸。"}),
                    }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doit"

    CATEGORY = "📜Firetheft AI Tools"

    def doit(self, image, scale_method, scale_factor, max_megapixels, chunk_frames, cache_backend, upscale_model_opt=None):
        if upscale_model_opt is None:
            image = image_upscale_on_pixel_space(image, scale_method, scale_factor)
        else:
            image = image_upscale_on_pixel_space_with_model_optimized(
                image, scale_method, upscale_model_opt, scale_factor, 
                max_megapixels, chunk_frames, cache_backend
            )
        return (image,)

NODE_CLASS_MAPPINGS = {
    "ImagePixelScaleNode": ImagePixelScaleNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePixelScaleNode": "Image Pixel Scale（图像像素空间缩放）"
}
