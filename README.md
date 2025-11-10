# ComfyUI-Sa2VA-XJ

Simple implementation of [ByteDance Sa2VA](https://github.com/bytedance/Sa2VA) nodes for ComfyUI.

## Overview

Sa2VA (Segment Anything 2 Video Assistant) is a multimodal large language model that combines SAM2 segmentation with vision-language understanding. This refined edition provides simple, maintainable nodes for image and video segmentation.

## Features

- ✅ **Two dedicated nodes**: Separate nodes for image and video processing
- ✅ **Configurable mask threshold**: Control mask quality with raw sigmoid probabilities (0.0-1.0, step 0.05)
- ✅ **8-bit quantization**: Save VRAM with proper vision component handling
- ✅ **Flash attention**: Optional acceleration for faster inference
- ✅ **Model unloading**: Free VRAM after inference (user-controllable)
- ✅ **Fail-fast errors**: Clear error messages, no silent failures

## Installation

### 1. Clone Repository

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/alexjx/ComfyUI-Sa2VA-XJ.git
cd ComfyUI-Sa2VA-XJ
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Optional Dependencies

**For 8-bit quantization (saves VRAM):**
```bash
pip install bitsandbytes
```

**For flash attention (faster inference):**
```bash
pip install flash-attn --no-build-isolation
```

### 4. Restart ComfyUI

## Requirements

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **transformers**: >= 4.57.0 (critical!)
- **CUDA**: 11.8+ (for GPU acceleration)
- **VRAM**:
  - 8GB+ for 2B-4B models
  - 16GB+ for 7B-8B models
  - 24GB+ for 14B models
  - Use 8-bit quantization if VRAM-limited

## Nodes

### Sa2VA Image Segmentation

Process single images with Sa2VA.

**Inputs:**
- `model_name`: Model to use (default: Qwen3-VL-4B)
- `image`: Input image (IMAGE type)
- `segmentation_prompt`: Description of what to segment (STRING)
- `threshold`: Binary threshold for masks (FLOAT, 0.0-1.0, step 0.05, default: 0.5)
- `use_8bit`: Enable 8-bit quantization (BOOLEAN, default: False)
- `use_flash_attn`: Enable flash attention (BOOLEAN, default: True)
- `unload`: Unload model after inference (BOOLEAN, default: True)

**Outputs:**
- `text_output`: Generated text description (STRING)
- `masks`: Segmentation masks (MASK, [B, H, W])

**Example Prompts:**
```
"Please segment the person in the image."
"Provide masks for all objects."
"Segment the car on the left side."
```

### Sa2VA Video Segmentation

Process video frames or image batches with Sa2VA.

**Inputs:**
- `model_name`: Model to use (default: Qwen3-VL-4B)
- `images`: Input frames (IMAGE type, batch)
- `segmentation_prompt`: Description of what to segment (STRING)
- `threshold`: Binary threshold for masks (FLOAT, 0.0-1.0, step 0.05, default: 0.7)
- `use_8bit`: Enable 8-bit quantization (BOOLEAN, default: False)
- `use_flash_attn`: Enable flash attention (BOOLEAN, default: True)
- `unload`: Unload model after inference (BOOLEAN, default: True)

**Outputs:**
- `text_output`: Generated video description (STRING)
- `masks`: Segmentation masks (MASK, [B, H, W])

**Example Prompts:**
```
"Segment the person throughout the video."
"Track and segment the moving car."
"Provide masks for all objects in this video sequence."
```

## Supported Models

| Model                           | Parameters | VRAM (fp16) | VRAM (8-bit) |
| ------------------------------- | ---------- | ----------- | ------------ |
| ByteDance/Sa2VA-InternVL3-2B    | 2B         | ~6GB        | ~4GB         |
| ByteDance/Sa2VA-Qwen2_5-VL-3B   | 3B         | ~8GB        | ~5GB         |
| **ByteDance/Sa2VA-Qwen3-VL-4B** | 4B         | ~10GB       | ~6GB         |
| ByteDance/Sa2VA-Qwen2_5-VL-7B   | 7B         | ~16GB       | ~10GB        |
| ByteDance/Sa2VA-InternVL3-8B    | 8B         | ~18GB       | ~11GB        |
| ByteDance/Sa2VA-InternVL3-14B   | 14B        | ~30GB       | ~18GB        |

**Recommended:** Sa2VA-Qwen3-VL-4B (best balance of quality and VRAM)

## Usage Tips

### Save VRAM
1. Enable **use_8bit** (saves ~40% VRAM)
2. Use **smaller models** (2B-4B)
3. Keep **unload = True** (default)
4. Disable **use_flash_attn** if not installed

### Improve Quality
1. Use **specific prompts**: "woman on the right" vs "person"
2. Use **descriptive text**: Sa2VA handles long prompts well
3. Adjust **threshold**:
   - Lower (0.3-0.4): More inclusive masks, captures low-confidence regions
   - Default (0.5): Balanced segmentation
   - Higher (0.6-0.7): Stricter masks, only high-confidence regions
4. Use **larger models**: 7B-14B for complex scenes

### Speed Up Inference
1. Enable **use_flash_attn** (requires flash-attn package)
2. Enable **use_8bit** (slight quality trade-off)
3. Use **smaller models** (2B-4B)

## Troubleshooting

### "transformers >= 4.57.0 required"
```bash
pip install transformers>=4.57.0 --upgrade
```
Restart ComfyUI after upgrade.

### "No module named 'qwen_vl_utils'"
```bash
pip install qwen_vl_utils
```

### "bitsandbytes required for 8-bit quantization"
```bash
pip install bitsandbytes
```
Or disable `use_8bit` in node settings.

### "Flash attention not available"
```bash
pip install flash-attn --no-build-isolation
```
Or disable `use_flash_attn` in node settings (not required).

### "CUDA Out of Memory"
1. Enable `use_8bit`
2. Use smaller model (2B or 4B)
3. Ensure `unload = True` (default)
4. Close other programs using VRAM

### "No masks generated" (all black masks)
1. Try more specific prompts
2. Adjust `threshold` (try 0.3-0.7 in 0.05 steps)
3. Check if objects are actually in the image
4. Try different model variant

### Model loads slowly
- First load downloads model (~8-30GB depending on size)
- Models cache in `~/.cache/huggingface/hub/`
- Subsequent loads are faster (loaded from cache)

## Examples

### Basic Image Segmentation
1. Load Image → Sa2VA Image Segmentation
2. Set prompt: "Please segment all objects"
3. Connect `masks` output to mask-compatible nodes

### Video Segmentation
1. Load Video frames → Sa2VA Video Segmentation
2. Set prompt: "Track the person throughout the video"
3. Connect outputs to downstream nodes

### Multi-object Segmentation
1. Use specific prompt: "Segment the red car and the person standing"
2. Adjust `threshold` to 0.5-0.6 for balanced results
3. Multiple masks will be output as batch

## Technical Details

### Mask Threshold Control

This implementation includes a **monkey-patched model** that returns raw sigmoid probabilities instead of binarized masks. This makes the `threshold` parameter fully functional:

- **Raw probabilities**: The model outputs continuous confidence values (0.0-1.0)
- **User control**: Your `threshold` parameter determines the binarization point
- **Fine-grained adjustment**: 0.05 step size allows precise control over mask quality

See `MASK_THRESHOLD_IMPLEMENTATION.md` for technical details on the implementation.

### Cache Location
Models are cached in the global HuggingFace cache:
- Linux: `~/.cache/huggingface/hub/`
- Windows: `C:\Users\USERNAME\.cache\huggingface\hub\`
- Respects `HF_HOME` environment variable

### 8-bit Quantization
Skips vision components to avoid errors:
- `visual`: Vision encoder (contains 4D conv layers)
- `grounding_encoder`: SAM2 grounding model
- `text_hidden_fcs`: Vision-to-text projection

Language model backbone is quantized (~70% of parameters).

### Model Unloading
When `unload = True` (default):
- Model is deleted from memory
- CUDA cache is cleared
- Garbage collection is forced
- VRAM is freed for other tasks

## Links

- [Sa2VA Paper](https://arxiv.org/abs/2501.04001)
- [Sa2VA GitHub](https://github.com/bytedance/Sa2VA)
- [Sa2VA Models on HuggingFace](https://huggingface.co/ByteDance)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## License

MIT

## Credits

- Based on [ByteDance Sa2VA](https://github.com/bytedance/Sa2VA)
- Inspired by [ComfyUI-Sa2VA](https://github.com/adambarbato/ComfyUI-Sa2VA)
