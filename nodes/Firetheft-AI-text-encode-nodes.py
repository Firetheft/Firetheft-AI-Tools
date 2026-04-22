import node_helpers
import comfy.utils
import math
import torch
import nodes  
import inspect  
import logging  

# 辅助函数：验证 resizing 索引
def validate_vl_resize_indexs(vl_resize_indexs_str, valid_length):
    try:
        indexes = [int(i)-1 for i in vl_resize_indexs_str.split(",")]
        indexes = list(set(indexes))
    except ValueError as e:
        raise ValueError(f"Invalid format for vl_resize_indexs: {e}")

    if not indexes:
        raise ValueError("vl_resize_indexs must not be empty")

    indexes = [idx for idx in indexes if 0 <= idx < valid_length]

    return indexes

class TextEncodeFlux2KleinImageEdit_Enhanced:
    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["pad", "center", "disabled"]
    target_sizes = [1024, 1344, 1536, 2048, 768, 512]
    target_vl_sizes = [392, 384]
    divisor_options = [8, 16, 32, 64]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "clip": ("CLIP", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": 
            {
                "vae": ("VAE", ),
                "image1": ("IMAGE", ),
                "mask1": ("MASK", ),
                "image2": ("IMAGE", ),
                "mask2": ("MASK", ),
                "image3": ("IMAGE", ),
                "mask3": ("MASK", ),
                "image4": ("IMAGE", ),
                "mask4": ("MASK", ),
                "image5": ("IMAGE", ),
                "mask5": ("MASK", ),
                "vl_resize_indexs": ("STRING", {"default": "1,2,3"}),
                "main_image_index": ("INT", {"default": 1, "max": 5, "min": 1}),
                "target_size": (s.target_sizes, {"default": 1024}),
                "target_vl_size": (s.target_vl_sizes, {"default": 384}),
                "upscale_method": (s.upscale_methods,),
                "crop_method": (s.crop_methods,),
                "divisible_by": (s.divisor_options, {"default": 8}),
                "use_tiled_vae": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "禁用"}), 
                "vae_tile_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}), 
                "enable_mask": ("BOOLEAN", {"default": True, "label_on": "启用", "label_off": "禁用"}), 
                # 已移除 Instruction 输入
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "CONDITIONING", "ANY")
    RETURN_NAMES = ("conditioning_with_full_ref", "latent", "image1", "image2", "image3", "image4", "image5", "conditioning_with_main_ref", "pad_info")
    FUNCTION = "encode"

    CATEGORY = "📜Firetheft AI Tools/文本编码"

    def encode(self, clip, prompt, vae=None, 
               image1=None, mask1=None, image2=None, mask2=None, image3=None, mask3=None,
               image4=None, mask4=None, image5=None, mask5=None, 
               vl_resize_indexs="1,2,3",
               main_image_index=1,
               target_size=1024, 
               target_vl_size=384,
               upscale_method="lanczos",
               crop_method="pad",
               divisible_by=8,
               # 已移除 instruction 参数
               use_tiled_vae=False, 
               vae_tile_size=512,
               enable_mask=True  
               ):
        
        target_size = int(target_size)
        target_vl_size = int(target_vl_size)
        divisible_by = int(divisible_by)
        div_float = float(divisible_by)
        resize_indexs = validate_vl_resize_indexs(vl_resize_indexs, 5)
        pad_info = {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "scale_by": 0,
            "original_width": 0,
            "original_height": 0
        }
        ref_latents = []
        ref_masks = []
        
        temp = [image1, image2, image3, image4, image5]
        # Filter out 1x1 placeholder images
        for i in range(len(temp)):
            img = temp[i]
            if img is not None and isinstance(img, torch.Tensor):
                if img.shape[1] <= 1 and img.shape[2] <= 1:
                    temp[i] = None

        masks_temp = [mask1, mask2, mask3, mask4, mask5] 

        if not enable_mask:
            masks_temp = [None] * 5
        
        images = []
        for i, image in enumerate(temp):
            image_dict = {
                "image": image,
                "mask": masks_temp[i],
                "vl_resize": False
            }
            if i in resize_indexs:
                image_dict['vl_resize'] = True
            images.append(image_dict)
            
        vl_resized_images = []
        
        vae_images = []
        vl_images = []
        
        # === 核心修改：移除 instruction 及其模版构建逻辑 ===
        # 原有的 template_prefix, instruction_content, llama_template 全部移除
        
        image_prompt = ""
        main_image_idx_user = main_image_index - 1 

        if vae is not None:
            for i, image_obj in enumerate(images):
                image = image_obj["image"]
                mask = image_obj["mask"]
                vl_resize = image_obj["vl_resize"]
                current_noise_mask = None 
                
                if image is not None:
                    samples = image.movedim(-1, 1)
                    _, c, _, _ = samples.shape
                    
                    mask_samples = None
                    if mask is not None:
                        mask_samples = mask.unsqueeze(1) 
                        if samples.shape[2:] != mask_samples.shape[2:]:
                            print(f"[TextEncodeFlux2Klein] 警告: 图像 {i+1} 和 蒙版 {i+1} 的尺寸不匹配。蒙版将被忽略。")
                            mask_samples = None
                            mask = None
                        else:
                            mask_samples = mask_samples.repeat(1, c, 1, 1)

                    current_total = (samples.shape[3] * samples.shape[2])
                    total = int(target_size * target_size)
                    scale_by = math.sqrt(total / current_total)
                    if crop_method == "pad":
                        crop = "center"
                        scaled_width = round(samples.shape[3] * scale_by)
                        scaled_height = round(samples.shape[2] * scale_by)
                        canvas_width = math.ceil(samples.shape[3] * scale_by / div_float) * divisible_by
                        canvas_height = math.ceil(samples.shape[2] * scale_by / div_float) * divisible_by
                        
                        canvas = torch.zeros(
                            (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                            dtype=samples.dtype,
                            device=samples.device
                        )
                        resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, upscale_method, crop)
                        resized_width = resized_samples.shape[3]
                        resized_height = resized_samples.shape[2]
                        
                        canvas[:, :, :resized_height, :resized_width] = resized_samples
                        
                        if mask is not None and mask_samples is not None:
                            mask_canvas = torch.zeros(
                                (samples.shape[0], c, canvas_height, canvas_width), 
                                dtype=samples.dtype,
                                device=samples.device
                            )
                            resized_mask_samples = comfy.utils.common_upscale(mask_samples, scaled_width, scaled_height, upscale_method, crop)
                            mask_canvas[:, :, :resized_height, :resized_width] = resized_mask_samples
                            current_noise_mask = mask_canvas[:, :1, :, :].squeeze(1)

                        if i == main_image_idx_user:
                            actual_resized_total = int(resized_width * resized_height)
                            if current_total == 0:
                                actual_scale_by = 1.0
                            else:
                                actual_scale_by = math.sqrt(actual_resized_total / current_total)

                            pad_info = {
                                "x": 0,
                                "y": 0,
                                "width": canvas_width - resized_width,
                                "height": canvas_height - resized_height,
                                "scale_by": round(1 / actual_scale_by, 5) if actual_scale_by != 0 else 1.0,
                                "original_width": samples.shape[3],
                                "original_height": samples.shape[2]
                            }

                        s = canvas
                    else:
                        width = round(samples.shape[3] * scale_by / div_float) * divisible_by
                        height = round(samples.shape[2] * scale_by / div_float) * divisible_by
                        crop = crop_method
                        s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                        
                        if mask is not None and mask_samples is not None:
                            m = comfy.utils.common_upscale(mask_samples, width, height, upscale_method, crop)
                            current_noise_mask = m[:, :1, :, :].squeeze(1)
                        
                    image = s.movedim(1, -1)
                    
                    pixels_for_vae = image[:, :, :, :3]
                    if use_tiled_vae:
                        try:
                            encoder = nodes.VAEEncodeTiled()
                            if 'overlap' in inspect.signature(encoder.encode).parameters:
                                encoded_latent = encoder.encode(vae, pixels_for_vae, vae_tile_size, overlap=64)[0]
                            else:
                                logging.warning("[TextEncodeFlux2Klein] 您的ComfyUI版本较旧。Tiled VAE overlap功能不可用。")
                                encoded_latent = encoder.encode(vae, pixels_for_vae, vae_tile_size)[0]
                        except Exception as e:
                            print(f"[TextEncodeFlux2Klein) VAE分块编码失败，回退到标准编码。错误: {e}")
                            encoded_latent = nodes.VAEEncode().encode(vae, pixels_for_vae)[0] 
                    else:
                        encoded_latent = nodes.VAEEncode().encode(vae, pixels_for_vae)[0] 
                    
                    ref_latents.append(encoded_latent["samples"])
                    ref_masks.append(current_noise_mask) 
                    
                    vae_images.append(image)
                    
                    if vl_resize:
                        total = int(target_vl_size * target_vl_size)
                        scale_by = math.sqrt(total / current_total)
                        
                        if crop_method == "pad":
                            crop = "center"
                            scaled_width = round(samples.shape[3] * scale_by)
                            scaled_height = round(samples.shape[2] * scale_by)
                            canvas_width = math.ceil(samples.shape[3] * scale_by / div_float) * divisible_by
                            canvas_height = math.ceil(samples.shape[2] * scale_by / div_float) * divisible_by
                            
                            canvas = torch.zeros(
                                (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                                dtype=samples.dtype,
                                device=samples.device
                            )
                            resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, upscale_method, crop)
                            resized_width = resized_samples.shape[3]
                            resized_height = resized_samples.shape[2]
                            
                            canvas[:, :, :resized_height, :resized_width] = resized_samples
                            s = canvas
                        else:
                            width = round(samples.shape[3] * scale_by / div_float) * divisible_by
                            height = round(samples.shape[2] * scale_by / div_float) * divisible_by
                            crop = crop_method
                            s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                        
                        image = s.movedim(1, -1)
                        vl_resized_images.append(image)

                    # 构建 Prompt (保持 Qwen 风格的 tag 结构，因为这通常是通用的，但去掉了 instruction)
                    image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                    vl_images.append(image)
                    
        # === 核心修改：Tokenize 时移除 llama_template ===
        # 原代码: tokens = clip.tokenize(image_prompt + prompt, images=vl_images, llama_template=llama_template)
        # 修改后: 仅传入文本和图像
        tokens = clip.tokenize(image_prompt + prompt, images=vl_images)
        
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning_full_ref = conditioning
        conditioning_with_main_ref = conditioning
        
        samples = torch.zeros(1, 4, 128, 128)
        final_noise_mask = None 
        
        if len(ref_latents) > 0:
            main_image_idx = main_image_index - 1
            if main_image_idx < 0 or main_image_idx >= len(ref_latents): 
                print("\n 自动修复 main_image_index 到第一个图像索引")
                main_image_idx = 0
                
            conditioning_full_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
            conditioning_with_main_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latents[main_image_idx]]}, append=True)
            samples = ref_latents[main_image_idx]
            final_noise_mask = ref_masks[main_image_idx]
        
        latent_out = {"samples": samples}
        if final_noise_mask is not None:
            latent_out["noise_mask"] = final_noise_mask
            
        if len(vae_images) < len(temp):
            vae_images.extend([None] * (len(temp) - len(vae_images)))
        image1, image2, image3, image4, image5 = vae_images
        
        return (conditioning_full_ref, latent_out, image1, image2, image3, image4, image5, conditioning_with_main_ref, pad_info)

class TextEncodeQwenImageEdit_Enhanced:
    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["pad", "center", "disabled"]
    target_sizes = [1024, 1344, 1536, 2048, 768, 512]
    target_vl_sizes = [392,384]
    vl_resize_indexs = [1,2,3]
    main_image_index = 1
    divisor_options = [8, 16, 32, 64]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "clip": ("CLIP", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": 
            {
                "vae": ("VAE", ),
                "image1": ("IMAGE", ),
                "mask1": ("MASK", ),
                "image2": ("IMAGE", ),
                "mask2": ("MASK", ),
                "image3": ("IMAGE", ),
                "mask3": ("MASK", ),
                "image4": ("IMAGE", ),
                "mask4": ("MASK", ),
                "image5": ("IMAGE", ),
                "mask5": ("MASK", ),
                "vl_resize_indexs": ("STRING", {"default": "1,2,3"}),
                "main_image_index": ("INT", {"default": 1, "max": 5, "min": 1}),
                "target_size": (s.target_sizes, {"default": 1024}),
                "target_vl_size": (s.target_vl_sizes, {"default": 384}),
                "upscale_method": (s.upscale_methods,),
                "crop_method": (s.crop_methods,),
                "divisible_by": (s.divisor_options, {"default": 8}),
                "use_tiled_vae": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "禁用"}), 
                "vae_tile_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}), 
                "instruction": ("STRING", {"multiline": True, "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."}),
                
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "CONDITIONING", "ANY")
    RETURN_NAMES = ("conditioning_with_full_ref", "latent", "image1", "image2", "image3", "image4", "image5", "conditioning_with_main_ref", "pad_info")
    FUNCTION = "encode"

    CATEGORY = "📜Firetheft AI Tools/文本编码"

    def encode(self, clip, prompt, vae=None, 
               image1=None, mask1=None, image2=None, mask2=None, image3=None, mask3=None,
               image4=None, mask4=None, image5=None, mask5=None, 
               vl_resize_indexs="1,2,3",
               main_image_index=1,
               target_size=1024, 
               target_vl_size=384,
               upscale_method="lanczos",
               crop_method="pad",
               divisible_by=8,
               instruction="",
               use_tiled_vae=False, 
               vae_tile_size=512
               ):
        
        target_size = int(target_size)
        target_vl_size = int(target_vl_size)
        divisible_by = int(divisible_by)
        div_float = float(divisible_by)
        resize_indexs = validate_vl_resize_indexs(vl_resize_indexs,5)
        pad_info = {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "scale_by": 0,
            "original_width": 0,
            "original_height": 0
        }
        ref_latents = []
        ref_masks = []
        
        temp = [image1, image2, image3, image4, image5]
        # Filter out 1x1 placeholder images
        for i in range(len(temp)):
            img = temp[i]
            if img is not None and isinstance(img, torch.Tensor):
                if img.shape[1] <= 1 and img.shape[2] <= 1:
                    temp[i] = None

        masks_temp = [mask1, mask2, mask3, mask4, mask5] 
        
        images = []
        for i, image in enumerate(temp):
            image_dict = {
                "image": image,
                "mask": masks_temp[i],
                "vl_resize": False
            }
            if i in resize_indexs:
                image_dict['vl_resize'] = True
            images.append(image_dict)
            
        vl_resized_images = []
        
        vae_images = []
        vl_images = []
        template_prefix = "<|im_start|>system\n"
        template_suffix = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        instruction_content = ""
        if instruction == "":
            instruction_content = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
        else:
            if template_prefix in instruction:
                instruction = instruction.split(template_prefix)[1]
            if template_suffix in instruction:
                instruction = instruction.split(template_suffix)[0]
            if "{}" in instruction:
                instruction = instruction.replace("{}", "")
            instruction_content = instruction
        llama_template = template_prefix + instruction_content + template_suffix
        image_prompt = ""
        
        main_image_idx_user = main_image_index - 1 

        if vae is not None:
            for i, image_obj in enumerate(images):
                image = image_obj["image"]
                mask = image_obj["mask"]
                vl_resize = image_obj["vl_resize"]
                current_noise_mask = None 
                
                if image is not None:
                    samples = image.movedim(-1, 1)
                    _, c, _, _ = samples.shape
                    
                    mask_samples = None
                    if mask is not None:
                        mask_samples = mask.unsqueeze(1) 
                        if samples.shape[2:] != mask_samples.shape[2:]:
                            print(f"[TextEncodeQwenEdit] 警告: 图像 {i+1} 和 蒙版 {i+1} 的尺寸不匹配。蒙版将被忽略。")
                            mask_samples = None
                            mask = None
                        else:
                            mask_samples = mask_samples.repeat(1, c, 1, 1)

                    current_total = (samples.shape[3] * samples.shape[2])
                    total = int(target_size * target_size)
                    scale_by = math.sqrt(total / current_total)
                    if crop_method == "pad":
                        crop = "center"
                        scaled_width = round(samples.shape[3] * scale_by)
                        scaled_height = round(samples.shape[2] * scale_by)
                        canvas_width = math.ceil(samples.shape[3] * scale_by / div_float) * divisible_by
                        canvas_height = math.ceil(samples.shape[2] * scale_by / div_float) * divisible_by
                        
                        canvas = torch.zeros(
                            (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                            dtype=samples.dtype,
                            device=samples.device
                        )
                        resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, upscale_method, crop)
                        resized_width = resized_samples.shape[3]
                        resized_height = resized_samples.shape[2]
                        
                        canvas[:, :, :resized_height, :resized_width] = resized_samples
                        
                        if mask is not None and mask_samples is not None:
                            mask_canvas = torch.zeros(
                                (samples.shape[0], c, canvas_height, canvas_width), 
                                dtype=samples.dtype,
                                device=samples.device
                            )
                            resized_mask_samples = comfy.utils.common_upscale(mask_samples, scaled_width, scaled_height, upscale_method, crop)
                            mask_canvas[:, :, :resized_height, :resized_width] = resized_mask_samples
                            current_noise_mask = mask_canvas[:, :1, :, :].squeeze(1)

                        if i == main_image_idx_user:
                            actual_resized_total = int(resized_width * resized_height)
                            if current_total == 0:
                                actual_scale_by = 1.0
                            else:
                                actual_scale_by = math.sqrt(actual_resized_total / current_total)

                            pad_info = {
                                "x": 0,
                                "y": 0,
                                "width": canvas_width - resized_width,
                                "height": canvas_height - resized_height,
                                "scale_by": round(1 / actual_scale_by, 5) if actual_scale_by != 0 else 1.0,
                                "original_width": samples.shape[3],
                                "original_height": samples.shape[2]
                            }

                        s = canvas
                    else:
                        width = round(samples.shape[3] * scale_by / div_float) * divisible_by
                        height = round(samples.shape[2] * scale_by / div_float) * divisible_by
                        crop = crop_method
                        s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                        
                        if mask is not None and mask_samples is not None:
                            m = comfy.utils.common_upscale(mask_samples, width, height, upscale_method, crop)
                            current_noise_mask = m[:, :1, :, :].squeeze(1)
                        
                    image = s.movedim(1, -1)
                    
                    pixels_for_vae = image[:, :, :, :3]
                    if use_tiled_vae:
                        try:
                            encoder = nodes.VAEEncodeTiled()
                            if 'overlap' in inspect.signature(encoder.encode).parameters:
                                encoded_latent = encoder.encode(vae, pixels_for_vae, vae_tile_size, overlap=64)[0]
                            else:
                                logging.warning("[TextEncodeQwenEdit] 您的ComfyUI版本较旧。Tiled VAE overlap功能不可用。")
                                encoded_latent = encoder.encode(vae, pixels_for_vae, vae_tile_size)[0]
                        except Exception as e:
                            print(f"[TextEncodeQwenEdit] VAE分块编码失败，回退到标准编码。错误: {e}")
                            encoded_latent = nodes.VAEEncode().encode(vae, pixels_for_vae)[0] 
                    else:
                        encoded_latent = nodes.VAEEncode().encode(vae, pixels_for_vae)[0] 
                    
                    ref_latents.append(encoded_latent["samples"])
                    ref_masks.append(current_noise_mask) 
                    
                    vae_images.append(image)
                    
                    if vl_resize:
                        total = int(target_vl_size * target_vl_size)
                        scale_by = math.sqrt(total / current_total)
                        
                        if crop_method == "pad":
                            crop = "center"
                            scaled_width = round(samples.shape[3] * scale_by)
                            scaled_height = round(samples.shape[2] * scale_by)
                            canvas_width = math.ceil(samples.shape[3] * scale_by / div_float) * divisible_by
                            canvas_height = math.ceil(samples.shape[2] * scale_by / div_float) * divisible_by
                            
                            canvas = torch.zeros(
                                (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                                dtype=samples.dtype,
                                device=samples.device
                            )
                            resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, upscale_method, crop)
                            resized_width = resized_samples.shape[3]
                            resized_height = resized_samples.shape[2]
                            
                            canvas[:, :, :resized_height, :resized_width] = resized_samples
                            s = canvas
                        else:
                            width = round(samples.shape[3] * scale_by / div_float) * divisible_by
                            height = round(samples.shape[2] * scale_by / div_float) * divisible_by
                            crop = crop_method
                            s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                        
                        image = s.movedim(1, -1)
                        vl_resized_images.append(image)

                    image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                    vl_images.append(image)
                    
        tokens = clip.tokenize(image_prompt + prompt, images=vl_images, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning_full_ref = conditioning
        conditioning_with_main_ref = conditioning
        
        samples = torch.zeros(1, 4, 128, 128)
        final_noise_mask = None 
        
        if len(ref_latents) > 0:
            main_image_idx = main_image_index - 1
            if main_image_idx < 0 or main_image_idx >= len(ref_latents): 
                print("\n 自动修复 main_image_index 到第一个图像索引")
                main_image_idx = 0
                
            conditioning_full_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
            conditioning_with_main_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latents[main_image_idx]]}, append=True)
            samples = ref_latents[main_image_idx]
            final_noise_mask = ref_masks[main_image_idx]
        
        latent_out = {"samples": samples}
        if final_noise_mask is not None:
            latent_out["noise_mask"] = final_noise_mask
            
        if len(vae_images) < len(temp):
            vae_images.extend([None] * (len(temp) - len(vae_images)))
        image1, image2, image3, image4, image5 = vae_images
        
        return (conditioning_full_ref, latent_out, image1, image2, image3, image4, image5, conditioning_with_main_ref, pad_info)

class CropWithPadInfo_Enhanced:
    upscale_methods = ["lanczos", "bicubic", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "pad_info": ("ANY", ),
                
                "enable_scaling": ("BOOLEAN", {
                    "default": True, 
                    "label_on": "启用缩放", 
                    "label_off": "仅裁切"
                }),
                
                "latent_upscale_factor": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.1, 
                    "max": 16.0,
                    "step": 0.1,
                    "round": 0.01,
                    "display": "number"
                }),
                
                "scale_method": (s.upscale_methods, {"default": "lanczos"}),
            },
            "optional": {
                "upscale_model_opt": ("UPSCALE_MODEL", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",) 
    FUNCTION = "crop_and_scale"
    CATEGORY = "📜Firetheft AI Tools/文本编码"

    def crop_and_scale(self, image, pad_info, enable_scaling=True, scale_method="lanczos", latent_upscale_factor=1.0, upscale_model_opt=None):
        
        x = pad_info.get("x", 0)
        y = pad_info.get("y", 0)
        width_padding = pad_info.get("width", 0)
        height_padding = pad_info.get("height", 0)
        scale_by = pad_info.get("scale_by", 1.0)

        # Safety: If input is a 1x1 placeholder, just return it without processing
        if image.shape[1] <= 1 and image.shape[2] <= 1:
            return (image,)

        scaled_width_padding = round(width_padding * latent_upscale_factor)
        scaled_height_padding = round(height_padding * latent_upscale_factor)

        img = image.movedim(-1, 1)

        original_content_width = img.shape[3] - scaled_width_padding
        original_content_height = img.shape[2] - scaled_height_padding

        original_content_width = max(1, min(original_content_width, img.shape[3]))
        original_content_height = max(1, min(original_content_height, img.shape[2]))
        x = max(0, min(x, img.shape[3] - 1))
        y = max(0, min(y, img.shape[2] - 1))

        cropped_img = img[:, :, y:original_content_height, x:original_content_width]
        
        if enable_scaling:
            
            if scale_by == 0.0 or latent_upscale_factor == 0.0:
                final_tensor = cropped_img
            else:
                _b, _c, current_h, current_w = cropped_img.shape
                original_width = pad_info.get("original_width", None)
                original_height = pad_info.get("original_height", None)

                if original_width is not None and original_height is not None:
                    target_w = original_width
                    target_h = original_height
                else:
                    final_scale_factor = scale_by / latent_upscale_factor
                    target_w = int(round(current_w * final_scale_factor))
                    target_h = int(round(current_h * final_scale_factor))

                if target_w < 1 or target_h < 1:
                     target_w = max(1, target_w)
                     target_h = max(1, target_h)
                
                if upscale_model_opt is not None:
                    pixels = cropped_img.movedim(1, -1)
                    
                    while pixels.shape[2] < target_w:
                        if "ImageUpscaleWithModel" in nodes.NODE_CLASS_MAPPINGS:
                            upscaler = nodes.NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
                            pixels = upscaler.upscale(upscale_model_opt, pixels)[0]
                        else:
                            print("[CropWithPadInfo-Enhanced] ImageUpscaleWithModel node not found, skipping model upscale.")
                            break
                             
                        new_current_w = pixels.shape[2]
                        if new_current_w == pixels.shape[2]: 
                            break

                    rescaled_img = nodes.ImageScale().upscale(pixels, scale_method, target_w, target_h, "disabled")[0]
                    final_tensor = rescaled_img.movedim(-1, 1)
                else:
                    rescaled_img_tensor = comfy.utils.common_upscale(cropped_img, target_w, target_h, scale_method, "disabled")
                    final_tensor = rescaled_img_tensor
        else:
            final_tensor = cropped_img

        final_image = final_tensor.movedim(1, -1)
        
        return (final_image,)

NODE_CLASS_MAPPINGS = {
    "TextEncodeFlux2KleinImageEdit_Enhanced": TextEncodeFlux2KleinImageEdit_Enhanced,
    "TextEncodeQwenImageEdit_Enhanced": TextEncodeQwenImageEdit_Enhanced,
    "CropWithPadInfo_Enhanced": CropWithPadInfo_Enhanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeFlux2KleinImageEdit_Enhanced": "Flux2Klein Image Edit（文本编码器）",
    "TextEncodeQwenImageEdit_Enhanced": "Qwen Image Edit（文本编码器）",
    "CropWithPadInfo_Enhanced": "Crop With Pad Info（填充裁剪）",
}
