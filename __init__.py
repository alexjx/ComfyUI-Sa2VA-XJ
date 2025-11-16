# ComfyUI-Sa2VA-RE - Refined Edition
# Simplified Sa2VA nodes for ComfyUI

import os
import folder_paths

from .sa2va_node import XJSa2VAImageSegmentation, XJSa2VAVideoSegmentation
from .sa2va_node_v2 import XJSa2VAImageSegmentationV2

# Register custom model directories for ComfyUI
models_dir = folder_paths.models_dir

# Register Sa2VA models directory
sa2va_dir = os.path.join(models_dir, "sa2va")
os.makedirs(sa2va_dir, exist_ok=True)
folder_paths.add_model_folder_path("sa2va", sa2va_dir)

# Register VITMatte models directory
vitmatte_dir = os.path.join(models_dir, "vitmatte")
os.makedirs(vitmatte_dir, exist_ok=True)
folder_paths.add_model_folder_path("vitmatte", vitmatte_dir)

NODE_CLASS_MAPPINGS = {
    "XJSa2VAImageSegmentation": XJSa2VAImageSegmentation,
    "XJSa2VAVideoSegmentation": XJSa2VAVideoSegmentation,
    "XJSa2VAImageSegmentationV2": XJSa2VAImageSegmentationV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJSa2VAImageSegmentation": "Sa2VA Image Segmentation",
    "XJSa2VAVideoSegmentation": "Sa2VA Video Segmentation",
    "XJSa2VAImageSegmentationV2": "Sa2VA Image Segmentation V2",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
