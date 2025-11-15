"""
Sa2VA Node V2 with VITMatte Post-Processing
Image segmentation only - no morphological operations
"""

import gc
import numpy as np
import torch
from PIL import Image

from .sa2va_node import Sa2VABase


class XJSa2VAImageSegmentationV2(Sa2VABase):
    """Sa2VA V2 node with VITMatte post-processing for professional alpha mattes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    [
                        "ByteDance/Sa2VA-Qwen3-VL-4B",
                        "ByteDance/Sa2VA-InternVL3-2B",
                        "ByteDance/Sa2VA-Qwen2_5-VL-3B",
                        "ByteDance/Sa2VA-Qwen2_5-VL-7B",
                        "ByteDance/Sa2VA-InternVL3-8B",
                        "ByteDance/Sa2VA-InternVL3-14B",
                    ],
                    {"default": "ByteDance/Sa2VA-Qwen3-VL-4B"},
                ),
                "image": (
                    "IMAGE",
                    {"tooltip": "Input image to segment. Should be in RGB format."},
                ),
                "segmentation_prompt": (
                    "STRING",
                    {
                        "default": "Please provide segmentation masks for all objects.",
                        "multiline": True,
                        "tooltip": "Text prompt describing what objects to segment in the image. Be specific for better results.",
                    },
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Threshold for converting probability masks to binary masks. Higher values create stricter masks.",
                    },
                ),
                "use_8bit": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use 8-bit quantization to reduce memory usage (requires bitsandbytes).",
                    },
                ),
                "use_flash_attn": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use flash attention for faster processing (requires flash-attn).",
                    },
                ),
                "unload": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Unload model from memory after processing to free VRAM.",
                    },
                ),
                # VITMatte Post-Processing Section
                "process_detail": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enable VITMatte post-processing for smooth edges and fine details (AI-powered, slower)",
                    },
                ),
                "detail_erode": (
                    "INT",
                    {
                        "default": 6,
                        "min": 1,
                        "max": 255,
                        "step": 1,
                        "tooltip": "Erosion kernel size for trimap generation. "
                                   "Creates inner boundary (definite foreground). Higher = tighter edge.",
                    },
                ),
                "detail_dilate": (
                    "INT",
                    {
                        "default": 6,
                        "min": 1,
                        "max": 255,
                        "step": 1,
                        "tooltip": "Dilation kernel size for trimap generation. "
                                   "Creates outer boundary (uncertain region). Higher = wider transition zone.",
                    },
                ),
                "black_point": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.01,
                        "max": 0.98,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "Histogram black point. Values below this become 0. "
                                   "Higher = more aggressive background cleanup, removes gray halos.",
                    },
                ),
                "white_point": (
                    "FLOAT",
                    {
                        "default": 0.99,
                        "min": 0.02,
                        "max": 0.99,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "Histogram white point. Values above this become 1. "
                                   "Higher = more solid foreground.",
                    },
                ),
                "max_megapixels": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.5,
                        "max": 10.0,
                        "step": 0.5,
                        "tooltip": "Max resolution for VITMatte processing. "
                                   "Images larger than this are downscaled. Higher = better quality but more VRAM.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "MASK")
    RETURN_NAMES = ("text_output", "masks")
    OUTPUT_TOOLTIPS = ("Text output from the model", "Generated segmentation masks")
    FUNCTION = "segment_image"
    CATEGORY = "Sa2VA"
    DESCRIPTION = (
        "Sa2VA V2 with VITMatte post-processing for professional-quality alpha mattes. "
        "Provides smooth edges and fine detail preservation for hair, fur, glass, and complex edges. "
        "No morphological operations - pure VITMatte refinement."
    )

    def segment_image(
        self,
        model_name,
        image,
        segmentation_prompt,
        threshold,
        use_8bit,
        use_flash_attn,
        unload,
        # VITMatte parameters
        process_detail=True,
        detail_erode=6,
        detail_dilate=6,
        black_point=0.15,
        white_point=0.99,
        max_megapixels=2.0,
    ):
        """Process single image with Sa2VA and VITMatte.

        Args:
            model_name: HuggingFace model identifier
            image: ComfyUI IMAGE tensor [B, H, W, C] or [H, W, C]
            segmentation_prompt: Text prompt for segmentation
            threshold: Threshold for binary masks
            use_8bit: Use 8-bit quantization
            use_flash_attn: Use flash attention
            unload: Unload model after inference
            process_detail: Enable VITMatte post-processing
            detail_erode: VITMatte trimap erosion size
            detail_dilate: VITMatte trimap dilation size
            black_point: Histogram black point
            white_point: Histogram white point
            max_megapixels: VITMatte max resolution

        Returns:
            Tuple of (text_output, masks)
        """
        if image is None:
            raise ValueError("No image provided")

        # Load model
        self.load_model(model_name, use_8bit, use_flash_attn)

        # Convert to PIL
        pil_image = self._tensor_to_pil(image)

        # Prepare input
        input_dict = {
            "image": pil_image,
            "text": f"<image>{segmentation_prompt}",
            "past_text": "",
            "mask_prompts": None,
            "processor": self.processor,
            "return_raw_masks": True,  # Get raw sigmoid probabilities
        }

        # Inference
        print("Processing image...")
        with torch.inference_mode():
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    output = self.model.predict_forward(**input_dict)
            else:
                output = self.model.predict_forward(**input_dict)

        text = output.get("prediction", "")
        masks = output.get("prediction_masks", [])

        print(f"✓ Generated {len(masks)} mask(s)")

        # Unload Sa2VA model immediately if requested (before VITMatte processing)
        if unload:
            print("Unloading Sa2VA model before VITMatte processing...")
            self._unload_model()

        # Convert masks WITH VITMatte processing (Sa2VA no longer in VRAM if unloaded)
        h, w = pil_image.size[1], pil_image.size[0]
        comfyui_masks, _ = self._convert_masks_to_comfyui_v2(
            masks,
            h,
            w,
            threshold,
            # VITMatte parameters
            process_detail=process_detail,
            detail_erode=detail_erode,
            detail_dilate=detail_dilate,
            black_point=black_point,
            white_point=white_point,
            max_megapixels=max_megapixels,
            original_image_pil=pil_image,
        )

        return (text, comfyui_masks)

    def _convert_masks_to_comfyui_v2(
        self,
        masks,
        height,
        width,
        threshold,
        # VITMatte parameters
        process_detail=True,
        detail_erode=6,
        detail_dilate=6,
        black_point=0.15,
        white_point=0.99,
        max_megapixels=2.0,
        original_image_pil=None,
    ):
        """Convert Sa2VA masks to ComfyUI format with VITMatte post-processing.

        Args:
            masks: List of numpy arrays from Sa2VA
            height: Target height
            width: Target width
            threshold: Binary threshold for masks
            process_detail: Enable VITMatte post-processing
            detail_erode: VITMatte trimap erode size
            detail_dilate: VITMatte trimap dilate size
            black_point: Histogram black point
            white_point: Histogram white point
            max_megapixels: VITMatte max resolution
            original_image_pil: PIL Image required for VITMatte

        Returns:
            Tuple of (MASK tensor [B, H, W], IMAGE tensor [B, H, W, 3])
        """
        # Initialize VITMatte processor if enabled
        vitmatte_processor = None
        if process_detail:
            if original_image_pil is None:
                print("Warning: VITMatte enabled but no image provided. Falling back to threshold.")
                process_detail = False
            else:
                try:
                    from .sa2va_vitmatte import VITMattePostProcessor

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    vitmatte_processor = VITMattePostProcessor(
                        detail_erode=detail_erode,
                        detail_dilate=detail_dilate,
                        black_point=black_point,
                        white_point=white_point,
                        max_megapixels=max_megapixels,
                        device=device,
                    )
                    print("VITMatte post-processing enabled")
                except ImportError as e:
                    print(f"Warning: VITMatte not available: {e}. Install with: pip install transformers>=4.57.0")
                    process_detail = False
                except Exception as e:
                    print(f"Warning: Failed to initialize VITMatte: {e}")
                    process_detail = False

        if not masks or len(masks) == 0:
            empty_mask = torch.zeros((1, height, width), dtype=torch.float32)
            empty_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
            return empty_mask, empty_image

        comfyui_masks = []
        image_tensors = []

        for mask in masks:
            if mask is None:
                continue

            try:
                # Convert to numpy if needed
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.detach().cpu().numpy()
                elif isinstance(mask, np.ndarray):
                    mask_np = mask.copy()
                else:
                    continue

                # Handle different dimensions
                if len(mask_np.shape) == 4:  # (B, C, H, W)
                    mask_np = mask_np[0, 0]
                elif len(mask_np.shape) == 3:
                    # Check if video format [N_frames, H, W]
                    if mask_np.shape[0] < min(mask_np.shape[1], mask_np.shape[2]) / 2:
                        # Process each frame with VITMatte
                        for frame_idx in range(mask_np.shape[0]):
                            frame_mask = mask_np[frame_idx]

                            # Handle NaN and inf
                            if np.any(np.isnan(frame_mask)) or np.any(np.isinf(frame_mask)):
                                frame_mask = np.nan_to_num(frame_mask, nan=0.0, posinf=1.0, neginf=0.0)

                            # Normalize to 0-1
                            mask_min, mask_max = frame_mask.min(), frame_mask.max()
                            if mask_max > mask_min:
                                frame_mask = (frame_mask - mask_min) / (mask_max - mask_min)
                            else:
                                frame_mask = np.ones_like(frame_mask) if mask_min > 0 else np.zeros_like(frame_mask)

                            # === VITMatte Post-Processing ===
                            if process_detail and vitmatte_processor is not None:
                                try:
                                    frame_mask = vitmatte_processor.process_mask(
                                        original_image_pil,
                                        frame_mask
                                    )
                                except Exception as e:
                                    print(f"Warning: VITMatte failed for frame {frame_idx}: {e}")

                            # Apply threshold
                            frame_mask = (frame_mask > threshold).astype(np.float32)

                            # Add to output
                            comfyui_masks.append(torch.from_numpy(frame_mask).float())
                            rgb_np = np.stack([frame_mask, frame_mask, frame_mask], axis=-1)
                            image_tensors.append(torch.from_numpy(rgb_np))

                        continue  # Skip normal processing

                    elif mask_np.shape[0] == 1:  # (1, H, W)
                        mask_np = mask_np[0]
                    elif mask_np.shape[2] == 1:  # (H, W, 1)
                        mask_np = mask_np[:, :, 0]
                    else:
                        mask_np = mask_np[:, :, 0]

                # Ensure 2D
                if len(mask_np.shape) != 2:
                    continue

                # Convert to float
                if mask_np.dtype == bool:
                    mask_np = mask_np.astype(np.float32)
                elif not np.issubdtype(mask_np.dtype, np.floating):
                    mask_np = mask_np.astype(np.float32)

                # Handle NaN and inf
                if np.any(np.isnan(mask_np)) or np.any(np.isinf(mask_np)):
                    mask_np = np.nan_to_num(mask_np, nan=0.0, posinf=1.0, neginf=0.0)

                # Normalize to 0-1
                mask_min, mask_max = mask_np.min(), mask_np.max()
                if mask_max > mask_min:
                    mask_np = (mask_np - mask_min) / (mask_max - mask_min)
                else:
                    mask_np = np.ones_like(mask_np) if mask_min > 0 else np.zeros_like(mask_np)

                # === VITMatte Post-Processing ===
                if process_detail and vitmatte_processor is not None:
                    try:
                        print("Applying VITMatte post-processing...")
                        mask_np = vitmatte_processor.process_mask(
                            original_image_pil,
                            mask_np
                        )
                    except Exception as e:
                        print(f"Warning: VITMatte failed: {e}. Falling back to threshold.")

                # Apply threshold
                mask_np = (mask_np > threshold).astype(np.float32)

                # Convert to ComfyUI MASK format
                comfyui_mask = torch.from_numpy(mask_np).float()
                comfyui_masks.append(comfyui_mask)

                # Convert to IMAGE format (RGB visualization)
                rgb_np = np.stack([mask_np, mask_np, mask_np], axis=-1)
                rgb_np = np.clip(rgb_np, 0.0, 1.0).astype(np.float32)
                image_tensors.append(torch.from_numpy(rgb_np))

            except Exception as e:
                print(f"Warning: Error processing mask: {e}")
                continue

        # Cleanup VITMatte model
        if vitmatte_processor is not None:
            vitmatte_processor.cleanup()

        # Handle empty results
        if not comfyui_masks:
            empty_mask = torch.zeros((1, height, width), dtype=torch.float32)
            empty_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
            return empty_mask, empty_image

        # Stack masks into batch
        try:
            final_masks = torch.stack(comfyui_masks, dim=0).float()
            final_images = torch.stack(image_tensors, dim=0).float()
        except Exception as e:
            print(f"Warning: Error stacking masks: {e}")
            empty_mask = torch.zeros((1, height, width), dtype=torch.float32)
            empty_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
            return empty_mask, empty_image

        return final_masks, final_images


NODE_CLASS_MAPPINGS = {
    "XJSa2VAImageSegmentationV2": XJSa2VAImageSegmentationV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJSa2VAImageSegmentationV2": "Sa2VA Image Segmentation V2 (VITMatte)",
}
