import importlib

node_list = [
    "Firetheft-AI-gemini-flash-node",
    "Firetheft-AI-color-palette-extractor-node",
    "Firetheft-AI-color-palette-picker-node",
    "Firetheft-AI-color-palette-transfer-node",
    "Firetheft-AI-resharpen-details-ksampler-node",
    "Firetheft-AI-multi-area-conditioning-node",
    "Firetheft-AI-qwen-llm-node",
    "Firetheft-AI-text-viewer-node",
    "Firetheft-AI-text-encode-nodes",
    "Firetheft-AI-face_detection_node",
    "Firetheft-AI-latent-pixel-scale-node",
    "Firetheft-AI-image-compare-node",
    "Firetheft-AI-jkass-sampler-node",
    "Firetheft-AI-audio-enhancer-node",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in node_list:
    imported_module = importlib.import_module(f".nodes.{module_name}", __name__)

    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **imported_module.NODE_DISPLAY_NAME_MAPPINGS}


WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]