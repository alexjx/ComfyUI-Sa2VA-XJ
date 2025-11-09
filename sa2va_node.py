"""
Sa2VA Nodes for ComfyUI - Refined Edition
Simple, clean implementation of Sa2VA segmentation nodes
"""

import torch
import numpy as np
from PIL import Image
import gc

# Check optional dependencies at module level
try:
    import bitsandbytes
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False

try:
    import flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


def check_transformers_version():
    """Check if transformers version is sufficient for Sa2VA models."""
    try:
        from transformers import __version__ as transformers_version
        version_parts = transformers_version.split(".")
        major, minor = int(version_parts[0]), int(version_parts[1])

        if major < 4 or (major == 4 and minor < 57):
            raise ImportError(
                f"Sa2VA requires transformers >= 4.57.0, found {transformers_version}. "
                f"Please run: pip install transformers>=4.57.0 --upgrade"
            )
    except Exception as e:
        raise ImportError(f"Error checking transformers version: {e}")


# Check version at import time
check_transformers_version()

# Import after version check
from transformers import AutoProcessor, AutoModel, BitsAndBytesConfig


class Sa2VABase:
    """Base class with shared functionality for Sa2VA nodes."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model = None

    def load_model(self, model_name, use_8bit_quantization, use_flash_attn):
        """Load Sa2VA model with specified configuration.

        Args:
            model_name: HuggingFace model identifier
            use_8bit_quantization: Whether to use 8-bit quantization
            use_flash_attn: Whether to use flash attention
        """
        # Check if model is already loaded
        if self.model is not None and self.current_model == model_name:
            print(f"✓ Model already loaded: {model_name}")
            return

        # Unload old model if switching
        if self.model is not None:
            print(f"Unloading previous model: {self.current_model}")
            self._unload_model()

        print(f"Loading Sa2VA: {model_name}")

        # Build model configuration
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        # Configure 8-bit quantization
        if use_8bit_quantization:
            if not HAS_BITSANDBYTES:
                raise ImportError(
                    "bitsandbytes required for 8-bit quantization. "
                    "Install with: pip install bitsandbytes"
                )

            # Skip vision components to avoid dimension errors
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=[
                    "visual",            # Vision encoder (4D conv layers)
                    "grounding_encoder", # SAM2 grounding model
                    "text_hidden_fcs",   # Vision-to-text projection
                ],
                llm_int8_threshold=6.0,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model_kwargs["quantization_config"] = quantization_config
            print("  Using 8-bit quantization (skipping vision components)")
        else:
            # Use bfloat16 if supported, else float16
            if torch.cuda.is_available():
                if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float16
            else:
                dtype = torch.float32
            model_kwargs["torch_dtype"] = dtype
            print(f"  Using dtype: {dtype}")

        # Configure flash attention
        if use_flash_attn:
            if HAS_FLASH_ATTN:
                model_kwargs["use_flash_attn"] = True
                print("  Flash attention: enabled")
            else:
                print("  Flash attention: not available (continuing without it)")

        try:
            # Load model (uses global HuggingFace cache by default)
            self.model = AutoModel.from_pretrained(model_name, **model_kwargs).eval()

            # Move to device if not using 8-bit quantization (8-bit handles device placement)
            if not use_8bit_quantization:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = self.model.to(device)
                print(f"  Model on device: {device}")

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
            )

            self.current_model = model_name
            print(f"✓ Model loaded successfully")

        except Exception as e:
            self.model = None
            self.processor = None
            self.current_model = None
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

    def _unload_model(self):
        """Unload model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        self.current_model = None

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

        # Force garbage collection
        gc.collect()

        print("✓ Model unloaded")

    def _tensor_to_pil(self, tensor):
        """Convert ComfyUI image tensor to PIL Image.

        Args:
            tensor: Image tensor [H, W, C] or [B, H, W, C]

        Returns:
            PIL Image in RGB mode
        """
        # Handle batch dimension
        if len(tensor.shape) == 4:
            tensor = tensor[0]

        # Convert to numpy
        if isinstance(tensor, torch.Tensor):
            image_np = tensor.detach().cpu().numpy()
        else:
            image_np = tensor

        # Convert to uint8
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8)

        # Handle channel-first format (C, H, W) -> (H, W, C)
        if len(image_np.shape) == 3 and image_np.shape[0] in [1, 3, 4]:
            if image_np.shape[0] < image_np.shape[1] and image_np.shape[0] < image_np.shape[2]:
                image_np = np.transpose(image_np, (1, 2, 0))

        # Handle alpha channel
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]

        # Convert grayscale to RGB
        if len(image_np.shape) == 3 and image_np.shape[2] == 1:
            image_np = np.repeat(image_np, 3, axis=2)

        # Create PIL image and ensure RGB mode
        pil_image = Image.fromarray(image_np)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        return pil_image

    def _convert_masks_to_comfyui(self, masks, height, width, threshold):
        """Convert Sa2VA masks to ComfyUI MASK and IMAGE formats.

        Args:
            masks: List of numpy arrays from Sa2VA
            height: Target height
            width: Target width
            threshold: Binary threshold for masks

        Returns:
            Tuple of (MASK tensor [B, H, W], IMAGE tensor [B, H, W, 3])
        """
        if not masks or len(masks) == 0:
            # Return empty masks
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
                    if mask_np.shape[0] == 1:  # (1, H, W)
                        mask_np = mask_np[0]
                    elif mask_np.shape[2] == 1:  # (H, W, 1)
                        mask_np = mask_np[:, :, 0]
                    elif mask_np.shape[0] < mask_np.shape[1] and mask_np.shape[0] < mask_np.shape[2]:
                        mask_np = mask_np[0]
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


class XJSa2VAImageSegmentation(Sa2VABase):
    """Sa2VA node for single image segmentation."""

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
                "image": ("IMAGE",),
                "segmentation_prompt": (
                    "STRING",
                    {
                        "default": "Please provide segmentation masks for all objects.",
                        "multiline": True,
                    },
                ),
                "mask_threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                    },
                ),
                "use_8bit_quantization": ("BOOLEAN", {"default": False}),
                "use_flash_attn": ("BOOLEAN", {"default": True}),
                "unload_model": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "MASK")
    RETURN_NAMES = ("text_output", "masks")
    FUNCTION = "segment_image"
    CATEGORY = "Sa2VA"

    def segment_image(
        self,
        model_name,
        image,
        segmentation_prompt,
        mask_threshold,
        use_8bit_quantization,
        use_flash_attn,
        unload_model,
    ):
        """Process single image with Sa2VA.

        Args:
            model_name: HuggingFace model identifier
            image: ComfyUI IMAGE tensor [B, H, W, C] or [H, W, C]
            segmentation_prompt: Text prompt for segmentation
            mask_threshold: Threshold for binary masks
            use_8bit_quantization: Use 8-bit quantization
            use_flash_attn: Use flash attention
            unload_model: Unload model after inference

        Returns:
            Tuple of (text_output, masks)
        """
        if image is None:
            raise ValueError("No image provided")

        # Load model
        self.load_model(model_name, use_8bit_quantization, use_flash_attn)

        # Convert to PIL
        pil_image = self._tensor_to_pil(image)

        # Prepare input
        input_dict = {
            "image": pil_image,
            "text": f"<image>{segmentation_prompt}",
            "past_text": "",
            "mask_prompts": None,
            "processor": self.processor,
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

        # Convert masks
        h, w = pil_image.size[1], pil_image.size[0]
        comfyui_masks, _ = self._convert_masks_to_comfyui(
            masks, h, w, mask_threshold
        )

        # Unload if requested
        if unload_model:
            self._unload_model()

        return (text, comfyui_masks)


class XJSa2VAVideoSegmentation(Sa2VABase):
    """Sa2VA node for video/batch segmentation."""

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
                "images": ("IMAGE",),
                "segmentation_prompt": (
                    "STRING",
                    {
                        "default": "Please provide segmentation masks for the objects in this video.",
                        "multiline": True,
                    },
                ),
                "mask_threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                    },
                ),
                "use_8bit_quantization": ("BOOLEAN", {"default": False}),
                "use_flash_attn": ("BOOLEAN", {"default": True}),
                "unload_model": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "MASK")
    RETURN_NAMES = ("text_output", "masks")
    FUNCTION = "segment_video"
    CATEGORY = "Sa2VA"

    def segment_video(
        self,
        model_name,
        images,
        segmentation_prompt,
        mask_threshold,
        use_8bit_quantization,
        use_flash_attn,
        unload_model,
    ):
        """Process video frames with Sa2VA.

        Args:
            model_name: HuggingFace model identifier
            images: ComfyUI IMAGE tensor [B, H, W, C] where B is number of frames
            segmentation_prompt: Text prompt for segmentation
            mask_threshold: Threshold for binary masks
            use_8bit_quantization: Use 8-bit quantization
            use_flash_attn: Use flash attention
            unload_model: Unload model after inference

        Returns:
            Tuple of (text_output, masks)
        """
        if images is None:
            raise ValueError("No images provided")

        # Load model
        self.load_model(model_name, use_8bit_quantization, use_flash_attn)

        # Convert batch to list of PIL images
        pil_frames = []
        batch_size = images.shape[0]

        print(f"Processing {batch_size} frames...")
        for i in range(batch_size):
            pil_frame = self._tensor_to_pil(images[i])
            pil_frames.append(pil_frame)

        # Prepare input for video mode
        input_dict = {
            "video": pil_frames,  # List of PIL images
            "text": f"<image>{segmentation_prompt}",
            "past_text": "",
            "mask_prompts": None,
            "processor": self.processor,
        }

        # Inference
        with torch.inference_mode():
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    output = self.model.predict_forward(**input_dict)
            else:
                output = self.model.predict_forward(**input_dict)

        text = output.get("prediction", "")
        masks = output.get("prediction_masks", [])

        print(f"✓ Generated {len(masks)} mask(s)")

        # Convert masks
        h, w = images.shape[1], images.shape[2]
        comfyui_masks, _ = self._convert_masks_to_comfyui(
            masks, h, w, mask_threshold
        )

        # Unload if requested
        if unload_model:
            self._unload_model()

        return (text, comfyui_masks)
