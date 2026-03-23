import torch
from nodes import PreviewImage

class ImageCompareNode(PreviewImage):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            },

            "hidden": {
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "compare"
    CATEGORY = "📜Firetheft AI Tools"
    OUTPUT_NODE = True

    def compare(self, image_a, image_b, prompt=None, extra_pnginfo=None):

        img_a = image_a[:1]
        img_b = image_b[:1]

        res_a = self.save_images(img_a, filename_prefix="Firetheft_compare_a", prompt=prompt, extra_pnginfo=extra_pnginfo)
        res_b = self.save_images(img_b, filename_prefix="Firetheft_compare_b", prompt=prompt, extra_pnginfo=extra_pnginfo)

        list_a = res_a['ui']['images']
        list_b = res_b['ui']['images']

        return { "ui": { "Firetheft_images": list_a + list_b } }

NODE_CLASS_MAPPINGS = {
    "ImageCompareNode": ImageCompareNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCompareNode": "Image Compare（图像对比）"
}