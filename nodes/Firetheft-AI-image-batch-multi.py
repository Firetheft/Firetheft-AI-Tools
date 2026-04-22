import torch
import comfy.utils
from comfy_api.latest import io

class FiretheftImageBatchMulti(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        # We define a fixed number of optional top-level inputs (5 slots).
        inputs = []
        for i in range(1, 6):
            inputs.append(io.Image.Input(f"image_{i}", optional=True, display_name=f"Image {i}"))
            
        return io.Schema(
            node_id="FiretheftImageBatchMulti",
            display_name="Firetheft Image Batch Multi",
            category="📜Firetheft AI Tools/Image",
            description="Smartly batch multiple images. Only connected inputs are included. Filters out 1x1 placeholders.",
            inputs=inputs,
            outputs=[
                io.Image.Output("images", display_name="images"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        # Collect all connected image tensors from kwargs
        valid_images = []
        for i in range(1, 6):
            img = kwargs.get(f"image_{i}")
            
            # 1. Basic validity check
            if img is not None and isinstance(img, torch.Tensor) and img.numel() > 0:
                # 2. Filter out 1x1 placeholder images (often used in some nodes as "empty" signals)
                if img.shape[1] <= 1 and img.shape[2] <= 1:
                    continue
                    
                # Standardize to 4D [B, H, W, C]
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)
                valid_images.append(img)
        
        if not valid_images:
            # Fallback for empty batch: return a 64x64 black image to avoid node error
            return io.NodeOutput(torch.zeros((1, 64, 64, 3)))

        # Use the first connected image as the reference for size
        ref_img = valid_images[0]
        ref_h, ref_w = ref_img.shape[1], ref_img.shape[2]
        
        processed_images = []
        for img in valid_images:
            curr_h, curr_w = img.shape[1], img.shape[2]
            
            # Auto-resize if dimensions don't match using lanczos
            if curr_h != ref_h or curr_w != ref_w:
                tmp = img.permute(0, 3, 1, 2)
                tmp = comfy.utils.common_upscale(tmp, ref_w, ref_h, "lanczos", "disabled")
                img = tmp.permute(0, 2, 3, 1) # back to [B, H, W, C]
            
            processed_images.append(img)
            
        # Concatenate all processed tensors along the batch dimension (0)
        out_batch = torch.cat(processed_images, dim=0)
        
        return io.NodeOutput(out_batch)

NODE_CLASS_MAPPINGS = {"FiretheftImageBatchMulti": FiretheftImageBatchMulti}
NODE_DISPLAY_NAME_MAPPINGS = {"FiretheftImageBatchMulti": "Firetheft Image Batch Multi"}
