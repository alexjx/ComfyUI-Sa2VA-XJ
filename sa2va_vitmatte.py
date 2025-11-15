"""
VITMatte Post-Processing for Sa2VA
Simple, focused implementation - VITMatte only
"""

import gc
import logging
import math
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class VITMattePostProcessor:
    """VITMatte-based mask refinement for Sa2VA."""

    def __init__(
        self,
        detail_erode: int = 6,
        detail_dilate: int = 6,
        black_point: float = 0.15,
        white_point: float = 0.99,
        max_megapixels: float = 2.0,
        device: str = "cuda",
    ):
        """Initialize VITMatte post-processor.

        Args:
            detail_erode: Erosion kernel size for trimap (inner boundary)
            detail_dilate: Dilation kernel size for trimap (outer boundary)
            black_point: Histogram black point (0.01-0.98)
            white_point: Histogram white point (0.02-0.99)
            max_megapixels: Max resolution for processing (0.5-10.0)
            device: "cuda" or "cpu"
        """
        self.detail_erode = detail_erode
        self.detail_dilate = detail_dilate
        self.black_point = black_point
        self.white_point = white_point
        self.max_megapixels = max_megapixels
        self.device = device

        self.model = None
        self.processor = None

    def generate_trimap(self, mask_np: np.ndarray) -> Image.Image:
        """Generate trimap from sigmoid mask.

        Args:
            mask_np: Raw sigmoid mask [H, W] in 0-1 range

        Returns:
            PIL Image with trimap (L mode):
                - 0 (black): Definite background
                - 128 (gray): Uncertain region (for VITMatte to decide)
                - 255 (white): Definite foreground

        Algorithm:
            1. Convert sigmoid (0-1) to binary (0/255) with threshold 0.5
            2. Erode to get definite foreground (shrink inward)
            3. Dilate to get possible foreground (expand outward)
            4. Uncertain region = dilated - eroded
        """
        # Convert to binary uint8 for OpenCV
        mask_binary = (mask_np > 0.5).astype(np.uint8) * 255

        # Create morphological kernels
        erode_kernel = np.ones((self.detail_erode, self.detail_erode), np.uint8)
        dilate_kernel = np.ones((self.detail_dilate, self.detail_dilate), np.uint8)

        # Generate trimap zones (5 iterations like LayerStyle)
        eroded = cv2.erode(mask_binary, erode_kernel, iterations=5)
        dilated = cv2.dilate(mask_binary, dilate_kernel, iterations=5)

        # Build trimap
        trimap = np.zeros_like(mask_binary)
        trimap[dilated == 255] = 128  # Uncertain region (gray)
        trimap[eroded == 255] = 255   # Definite foreground (white)
        # Background stays 0 (black)

        return Image.fromarray(trimap).convert('L')

    def process_mask(
        self,
        image_pil: Image.Image,
        mask_np: np.ndarray
    ) -> np.ndarray:
        """Process mask with VITMatte.

        Args:
            image_pil: Original RGB image as PIL Image
            mask_np: Raw sigmoid mask [H, W] in 0-1 range

        Returns:
            Refined alpha matte [H, W] in 0-1 range

        Pipeline:
            1. Generate trimap from sigmoid mask
            2. Resize if image too large (memory management)
            3. Run VITMatte AI inference
            4. Upscale back to original size
            5. Apply histogram remapping
        """
        # Load model (lazy loading)
        self._load_model()

        # Ensure RGB mode
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')

        # Step 1: Generate trimap
        trimap_pil = self.generate_trimap(mask_np)

        # Step 2: Resolution management (from LayerStyle)
        width, height = image_pil.size
        max_pixels = self.max_megapixels * 1_048_576

        if width * height > max_pixels:
            # Downscale for processing
            ratio = width / height
            target_width = int(math.sqrt(ratio * max_pixels))
            target_height = int(target_width / ratio)

            image_resized = image_pil.resize(
                (target_width, target_height),
                Image.BILINEAR
            )
            trimap_resized = trimap_pil.resize(
                (target_width, target_height),
                Image.BILINEAR
            )
        else:
            image_resized = image_pil
            trimap_resized = trimap_pil

        # Step 3: VITMatte inference
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        inputs = self.processor(
            images=image_resized,
            trimaps=trimap_resized,
            return_tensors="pt"
        )

        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            predictions = self.model(**inputs).alphas

        # Explicit cleanup of intermediate tensors
        del inputs

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Step 4: Convert to numpy and remove padding
        alpha_np = predictions[0, 0].cpu().numpy()

        # VITMatte works in 32px tiles - crop to actual size
        alpha_np = alpha_np[:image_resized.height, :image_resized.width]

        # Step 5: Upscale if downscaled
        if width * height > max_pixels:
            alpha_pil = Image.fromarray((alpha_np * 255).astype(np.uint8))
            alpha_pil = alpha_pil.resize((width, height), Image.BILINEAR)
            alpha_np = np.array(alpha_pil).astype(np.float32) / 255.0

        # Step 6: Histogram remapping
        alpha_np = self.histogram_remap(alpha_np)

        return alpha_np

    def histogram_remap(self, alpha_np: np.ndarray) -> np.ndarray:
        """Apply histogram remapping for edge cleanup.

        Args:
            alpha_np: Alpha matte [H, W] in 0-1 range

        Returns:
            Remapped alpha with enhanced contrast

        Effect:
            - Values < black_point → 0 (remove gray halos)
            - Values > white_point → 1 (solidify foreground)
            - Values between → linearly stretched

        Formula (from LayerStyle):
            bp = min(black_point, white_point - 0.001)
            scale = 1 / (white_point - bp)
            output = clip((input - bp) * scale, 0, 1)
        """
        bp = min(self.black_point, self.white_point - 0.001)
        scale = 1.0 / (self.white_point - bp)
        remapped = np.clip((alpha_np - bp) * scale, 0.0, 1.0)
        return remapped

    def _load_model(self):
        """Load VITMatte model (lazy loading)."""
        if self.model is not None:
            return

        from transformers import VitMatteImageProcessor, VitMatteForImageMatting

        model_name = "hustvl/vitmatte-small-composition-1k"

        self.processor = VitMatteImageProcessor.from_pretrained(
            model_name,
        )

        self.model = VitMatteForImageMatting.from_pretrained(
            model_name,
        )

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        logger.info(f"VITMatte model loaded on {device}")

    def cleanup(self):
        """Unload model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        gc.collect()
