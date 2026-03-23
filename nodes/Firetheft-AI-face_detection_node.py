import cv2
import numpy as np
import torch
import logging
import os
from typing import Tuple, List, Optional, Union
import folder_paths
import urllib.request

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Model Management - Focus on Human and Anime Faces
try:
    models_dir = os.path.abspath(os.path.join(folder_paths.get_folder_paths('checkpoints')[0], ".."))
    ULTRALYTICS_MODEL_DIR = os.path.join(models_dir, "ultralytics", "bbox")
except Exception:
    ULTRALYTICS_MODEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "models", "ultralytics", "bbox")

os.makedirs(ULTRALYTICS_MODEL_DIR, exist_ok=True)

YOLO_MODELS = {}
# YOLO Face Models (Human & Anime)
YOLO_FACE_MODELS = [
    "face.pt",
    "anime-face.pt"
]

FACE_MODEL_URLS = {
    "face.pt": "https://modelscope.cn/models/disambo/Tools/resolve/master/models/face.pt",
    "anime-face.pt": "https://modelscope.cn/models/disambo/Tools/resolve/master/models/anime-face.pt"
}

def get_face_yolo_model(model_name: str):
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("Ultralytics library is required for YOLO detection.")
    
    model_path = os.path.join(ULTRALYTICS_MODEL_DIR, model_name)
    if model_name not in YOLO_MODELS:
        if not os.path.exists(model_path):
            if model_name in FACE_MODEL_URLS:
                url = FACE_MODEL_URLS[model_name]
                try:
                    logger.info(f"Downloading model {model_name} from {url}...")
                    urllib.request.urlretrieve(url, model_path)
                except Exception as e:
                    logger.error(f"Download failed for {model_name}: {e}")
                    return None
        try:
            YOLO_MODELS[model_name] = YOLO(model_path)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None
    return YOLO_MODELS[model_name]

class FaceDetectionNode:
    """
    Firetheft Face Detection - Human and Anime Face detection node.
    Outputs: 
    - faces: Cropped face images (Batch)
    - face_mask: Full-frame aligned mask (Matching input size)
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "yolo_model": (YOLO_FACE_MODELS, {"default": "yolov12m-face.pt"}),
                "detection_threshold": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 1.0, "step": 0.05}),
                "min_face_size": ("INT", {"default": 32, "min": 8, "max": 1024, "step": 8}),
                "padding": ("INT", {"default": 32, "min": 0, "max": 1024, "step": 8}),
                "output_mode": (["largest_face", "all_faces"], {"default": "all_faces"}),
            },
            "optional": {
                "face_output_format": (["individual", "strip"], {"default": "individual"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("faces", "face_mask")
    FUNCTION = "detect_and_crop"
    CATEGORY = "📜Firetheft AI Tools"

    def detect_and_crop(self, image: torch.Tensor, yolo_model: str, 
                      detection_threshold: float, min_face_size: int, padding: int, 
                      output_mode: str, face_output_format: str = "individual"):
        
        if not ULTRALYTICS_AVAILABLE:
            logger.error("Ultralytics not installed.")
            return (torch.zeros((1, 512, 512, 3)), torch.zeros((1, 64, 64)))

        B, H, W, C = image.shape
        images_np = []
        for i in range(B):
            img = (image[i].cpu().numpy() * 255).astype(np.uint8)
            images_np.append(img)

        detector = get_face_yolo_model(yolo_model)
        if not detector:
            return (torch.zeros((1, 512, 512, 3)), torch.zeros((1, H, W)))

        all_crops = []
        # Full frame mask initialized to black (0)
        full_mask = torch.zeros((B, H, W))

        for idx, img_np in enumerate(images_np):
            boxes = self._detect_yolo(img_np, detector, detection_threshold, min_size=min_face_size)
            if not boxes:
                continue

            current_crops = []
            current_boxes = [] # To store individual boxes for mask generation

            for x, y, w, h in boxes:
                # 1. Image Crop (for faces output)
                crop = self._crop_with_padding(img_np, x, y, w, h, padding)
                current_crops.append(crop)
                current_boxes.append((x, y, w, h))
                
            if output_mode == "largest_face":
                idx_largest = np.argmax([c.shape[0] * c.shape[1] for c in current_crops])
                all_crops.append(current_crops[idx_largest])
                # Draw only the largest face box on the full mask
                lx, ly, lw, lh = current_boxes[idx_largest]
                full_mask[idx, ly:ly+lh, lx:lx+lw] = 1.0
            else:
                all_crops.extend(current_crops)
                # Draw all face boxes on the full mask
                for bx, by, bw, bh in current_boxes:
                    full_mask[idx, by:by+bh, bx:bx+bw] = 1.0

        if not all_crops:
            logger.warning("No faces detected.")
            return (torch.zeros((1, 512, 512, 3)), torch.zeros((1, H, W)))

        # Format Crops Output
        if output_mode == "all_faces" and face_output_format == "strip":
            max_h = min(1024, max(c.shape[0] for c in all_crops))
            resized_crops = []
            for c in all_crops:
                nw = int(max_h * (c.shape[1] / c.shape[0]))
                resized_crops.append(cv2.resize(c, (nw, max_h), interpolation=cv2.INTER_LANCZOS4))
            
            combined_img = np.hstack(resized_crops)
            output_img = torch.from_numpy(combined_img.astype(np.float32) / 255.0).unsqueeze(0)
        else:
            # Individual frames (Standard Batch)
            target_h = min(1024, max(c.shape[0] for c in all_crops))
            target_w = min(1024, max(c.shape[1] for c in all_crops))
            
            final_crops = []
            for c in all_crops:
                final_crops.append(cv2.resize(c, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4))
            
            output_img = torch.from_numpy(np.stack(final_crops, axis=0).astype(np.float32) / 255.0)

        return (output_img, full_mask)

    def _detect_yolo(self, img_np, detector, threshold, min_size):
        boxes = []
        try:
            results = detector(img_np, verbose=False)
            if results and results[0].boxes:
                for box in results[0].boxes:
                    if box.conf.item() >= threshold:
                        c = box.xyxy[0].cpu().numpy().astype(int)
                        w, h = c[2] - c[0], c[3] - c[1]
                        if w >= min_size and h >= min_size:
                            boxes.append((c[0], c[1], w, h))
        except Exception as e:
            logger.error(f"Detection error: {e}")
        return boxes

    def _crop_with_padding(self, img_np, x, y, w, h, p):
        H, W = img_np.shape[:2]
        x1, y1 = max(0, x - p), max(0, y - p)
        x2, y2 = min(W, x + w + p), min(H, y + h + p)
        return img_np[y1:y2, x1:x2]

# Registration
NODE_CLASS_MAPPINGS = {
    "FaceDetectionNode": FaceDetectionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetectionNode": "Face Detection（人脸识别裁剪）"
}