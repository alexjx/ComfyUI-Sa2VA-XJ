"""
Sa2VA Nodes for ComfyUI - Refined Edition
Simple, clean implementation of Sa2VA segmentation nodes
"""

import gc
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig


def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    """Extract hidden states for [SEG] tokens."""
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    if n_out == 0:
        return hidden_states[0:0]
    return hidden_states[-n_out:][seg_mask]


def predict_forward_with_raw_masks(
    self,
    image=None,
    video=None,
    text=None,
    past_text="",
    mask_prompts=None,
    tokenizer=None,
    processor=None,
    return_raw_masks=False,  # NEW PARAMETER
):
    """
    Modified predict_forward that can return raw sigmoid probabilities.

    This is a monkey-patched version of the original HuggingFace model's
    predict_forward method. The key difference is the addition of the
    'return_raw_masks' parameter which, when True, returns raw sigmoid
    probabilities instead of binarized masks.

    Args:
        return_raw_masks: If True, return raw sigmoid probabilities (0-1)
                         If False, apply > 0.5 threshold (original behavior)
    """
    # Call the original method implementation
    assert processor is not None
    self.processor = processor
    self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids("[SEG]")
    text = text.replace("<image>", "")

    if image is None and video is None and "<image>" not in past_text:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": past_text + text},
                ],
            }
        ]
        processsed_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        mm_inputs = self.processor(
            text=[processsed_text],
            images=None,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        mm_inputs = mm_inputs.to(self.device)
        ret_masks = []
    else:
        input_dict = {}
        if video is not None:
            from qwen_vl_utils import process_vision_info

            extra_pixel_values = []
            content = []
            ori_image_size = video[0].size
            for frame_idx, frame_image in enumerate(video):
                g_image = np.array(frame_image)
                g_image = self.extra_image_processor.apply_image(g_image)
                g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                extra_pixel_values.append(g_image)
                if frame_idx < 5:
                    content.append({"type": "image", "image": frame_image})

            content.append({"type": "text", "text": text})
            messages = [{"role": "user", "content": content}]

            processsed_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            mm_inputs = self.processor(
                text=[processsed_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            mm_inputs = mm_inputs.to(self.device)
            g_pixel_values = torch.stack(
                [
                    self.grounding_encoder.preprocess_image(pixel)
                    for pixel in extra_pixel_values
                ]
            ).to(self.torch_dtype)
            num_frames = min(5, len(video))
        else:
            from qwen_vl_utils import process_vision_info

            ori_image_size = image.size
            g_image = np.array(image)
            g_image = self.extra_image_processor.apply_image(g_image)
            g_pixel_values = (
                torch.from_numpy(g_image)
                .permute(2, 0, 1)
                .contiguous()
                .to(self.torch_dtype)
            )
            extra_pixel_values = [g_pixel_values]
            g_pixel_values = torch.stack(
                [
                    self.grounding_encoder.preprocess_image(pixel)
                    for pixel in extra_pixel_values
                ]
            ).to(self.torch_dtype)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text},
                    ],
                }
            ]
            processsed_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            mm_inputs = self.processor(
                text=[processsed_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            mm_inputs = mm_inputs.to(self.device)
            num_frames = 1

        input_dict["g_pixel_values"] = g_pixel_values
        ret_masks = []

    generate_output = self.model.generate(
        **mm_inputs,
        max_new_tokens=2048,
        do_sample=False,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )

    generate_output_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(mm_inputs.input_ids, generate_output.sequences)
    ]
    predict = self.processor.batch_decode(
        generate_output_trimmed, skip_special_tokens=False
    )[0].strip()

    if image is None and video is None and "<image>" not in past_text:
        return {"prediction": predict, "prediction_masks": ret_masks}

    # Extract segmentation masks
    hidden_states = generate_output.hidden_states
    last_hidden_states = [item[-1][0] for item in hidden_states]
    last_hidden_states = torch.cat(last_hidden_states, dim=0)
    seg_hidden_states = get_seg_hidden_states(
        last_hidden_states, generate_output.sequences[0][:-1], seg_id=self.seg_token_idx
    )
    all_seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)

    for seg_hidden_states in all_seg_hidden_states:
        seg_hidden_states = seg_hidden_states.unsqueeze(0)
        g_pixel_values = input_dict["g_pixel_values"]
        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
        pred_masks = self.grounding_encoder.language_embd_inference(
            sam_states, [seg_hidden_states] * num_frames
        )
        w, h = ori_image_size
        masks = F.interpolate(
            pred_masks, size=(h, w), mode="bilinear", align_corners=False
        )
        masks = masks[:, 0]
        masks = masks.sigmoid()  # Apply sigmoid to get probabilities

        # THIS IS THE KEY MODIFICATION:
        if not return_raw_masks:
            # Original behavior: binarize with hardcoded 0.5
            masks = masks > 0.5
        # else: return raw sigmoid probabilities (0-1 float values)

        masks = masks.cpu().numpy()
        ret_masks.append(masks)

    return {"prediction": predict, "prediction_masks": ret_masks}


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
                    "visual",  # Vision encoder (4D conv layers)
                    "grounding_encoder",  # SAM2 grounding model
                    "text_hidden_fcs",  # Vision-to-text projection
                ],
                llm_int8_threshold=6.0,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model_kwargs["quantization_config"] = quantization_config
            print("  Using 8-bit quantization (skipping vision components)")
        else:
            # Use bfloat16 if supported, else float16
            if torch.cuda.is_available():
                if (
                    hasattr(torch.cuda, "is_bf16_supported")
                    and torch.cuda.is_bf16_supported()
                ):
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

            # Monkey-patch the model's predict_forward method to support raw masks
            print("  Patching model to support configurable mask threshold...")
            self.model.predict_forward = MethodType(
                predict_forward_with_raw_masks, self.model
            )

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
            if (
                image_np.shape[0] < image_np.shape[1]
                and image_np.shape[0] < image_np.shape[2]
            ):
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
                    elif (
                        mask_np.shape[0] < mask_np.shape[1]
                        and mask_np.shape[0] < mask_np.shape[2]
                    ):
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

                # Normalize to 0-1 (this now actually matters for raw sigmoid values!)
                # Raw sigmoid values are already in 0-1 range, but we ensure it here
                mask_min, mask_max = mask_np.min(), mask_np.max()
                if mask_max > mask_min:
                    mask_np = (mask_np - mask_min) / (mask_max - mask_min)
                else:
                    mask_np = (
                        np.ones_like(mask_np)
                        if mask_min > 0
                        else np.zeros_like(mask_np)
                    )

                # Apply user-configurable threshold (NOW THIS IS ACTUALLY USEFUL!)
                # Raw masks are sigmoid probabilities (0-1), so threshold controls binarization
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
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
                "use_8bit": ("BOOLEAN", {"default": False}),
                "use_flash_attn": ("BOOLEAN", {"default": True}),
                "unload": ("BOOLEAN", {"default": True}),
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
        threshold,
        use_8bit,
        use_flash_attn,
        unload,
    ):
        """Process single image with Sa2VA.

        Args:
            model_name: HuggingFace model identifier
            image: ComfyUI IMAGE tensor [B, H, W, C] or [H, W, C]
            segmentation_prompt: Text prompt for segmentation
            threshold: Threshold for binary masks
            use_8bit: Use 8-bit quantization
            use_flash_attn: Use flash attention
            unload: Unload model after inference

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

        # Convert masks
        h, w = pil_image.size[1], pil_image.size[0]
        comfyui_masks, _ = self._convert_masks_to_comfyui(masks, h, w, threshold)

        # Unload if requested
        if unload:
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
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
                "use_8bit": ("BOOLEAN", {"default": False}),
                "use_flash_attn": ("BOOLEAN", {"default": True}),
                "unload": ("BOOLEAN", {"default": True}),
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
        threshold,
        use_8bit,
        use_flash_attn,
        unload,
    ):
        """Process video frames with Sa2VA.

        Args:
            model_name: HuggingFace model identifier
            images: ComfyUI IMAGE tensor [B, H, W, C] where B is number of frames
            segmentation_prompt: Text prompt for segmentation
            threshold: Threshold for binary masks
            use_8bit: Use 8-bit quantization
            use_flash_attn: Use flash attention
            unload: Unload model after inference

        Returns:
            Tuple of (text_output, masks)
        """
        if images is None:
            raise ValueError("No images provided")

        # Load model
        self.load_model(model_name, use_8bit, use_flash_attn)

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
            "return_raw_masks": True,  # Get raw sigmoid probabilities
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
        comfyui_masks, _ = self._convert_masks_to_comfyui(masks, h, w, threshold)

        # Unload if requested
        if unload:
            self._unload_model()

        return (text, comfyui_masks)
