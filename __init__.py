# ComfyUI-Sa2VA-RE - Refined Edition
# Simplified Sa2VA nodes for ComfyUI

from .sa2va_node import XJSa2VAImageSegmentation, XJSa2VAVideoSegmentation

NODE_CLASS_MAPPINGS = {
    "XJSa2VAImageSegmentation": XJSa2VAImageSegmentation,
    "XJSa2VAVideoSegmentation": XJSa2VAVideoSegmentation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJSa2VAImageSegmentation": "Sa2VA Image Segmentation",
    "XJSa2VAVideoSegmentation": "Sa2VA Video Segmentation",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
