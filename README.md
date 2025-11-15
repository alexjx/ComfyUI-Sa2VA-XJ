# ComfyUI-Sa2VA-XJ

Simple implementation of [ByteDance Sa2VA](https://github.com/bytedance/Sa2VA) nodes for ComfyUI.

## Features

- ✅ **Three nodes**: Image V1, Image V2, and Video processing
- ✅ **VITMatte post-processing (V2)**: AI-powered alpha matting
- ✅ **Configurable mask threshold**: Control mask quality (0.0-1.0, step 0.05)
- ✅ **Morphological operations (V1)**: Opening, closing, erode, dilate
- ✅ **8-bit quantization**: Save VRAM
- ✅ **Flash attention**: Faster inference
- ✅ **Model unloading**: Free VRAM after inference

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/alexjx/ComfyUI-Sa2VA-XJ.git
cd ComfyUI-Sa2VA-XJ
pip install -r requirements.txt
```

**Optional:**
```bash
pip install bitsandbytes              # For 8-bit quantization
pip install flash-attn --no-build-isolation  # For flash attention
pip install opencv-python              # For morphological operations
```

Restart ComfyUI after installation.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers >= 4.57.0
- CUDA 11.8+ (GPU)
- VRAM: 8GB+ (2B-4B models), 16GB+ (7B-8B models), 24GB+ (14B models)

## Nodes

### Sa2VA Image Segmentation (V1)

**Inputs:**
- `model_name`: Model selection (default: Qwen3-VL-4B)
- `image`: Input image
- `segmentation_prompt`: Text description
- `threshold`: Binary threshold (0.0-1.0, default: 0.5)
- `use_8bit`: 8-bit quantization (default: True)
- `use_flash_attn`: Flash attention (default: True)
- `unload`: Unload model after inference (default: True)
- `morph`: Morphological operation (none/opening/closing/erode/dilate)
- `erode_kernel`, `dilate_kernel`, `iterations`: Morphology parameters

**Outputs:**
- `text_output`: Text description
- `masks`: Segmentation masks

### Sa2VA Image Segmentation V2

VITMatte-based post-processing for smooth alpha mattes.

**Inputs:**
- `model_name`: Model selection (default: Qwen3-VL-4B)
- `image`: Input image
- `segmentation_prompt`: Text description
- `threshold`: Binary threshold (0.0-1.0, default: 0.5)
- `use_8bit`: 8-bit quantization (default: True)
- `use_flash_attn`: Flash attention (default: True)
- `unload`: Unload model after inference (default: True)
- `process_detail`: Enable VITMatte (default: True)
- `detail_erode`: Trimap erosion size (1-255, default: 6)
- `detail_dilate`: Trimap dilation size (1-255, default: 6)
- `black_point`: Histogram black point (0.01-0.98, default: 0.15)
- `white_point`: Histogram white point (0.02-0.99, default: 0.99)
- `max_megapixels`: Max resolution (0.5-10.0, default: 2.0)

**Outputs:**
- `text_output`: Text description
- `masks`: Alpha mattes

**V1 vs V2:**
- V1: Fast, morphological operations, solid objects
- V2: Slower, VITMatte refinement, hair/fur/glass/complex edges

### Sa2VA Video Segmentation

Process video frames or image batches.

**Inputs:**
- `model_name`: Model selection
- `images`: Input frames (batch)
- `segmentation_prompt`: Text description
- `threshold`: Binary threshold (0.0-1.0, default: 0.7)
- `use_8bit`, `use_flash_attn`, `unload`: Same as V1
- `morph`, `erode_kernel`, `dilate_kernel`, `iterations`: Morphology parameters

**Outputs:**
- `text_output`: Video description
- `masks`: Segmentation masks for all frames

## Supported Models

| Model | Parameters | VRAM (fp16) | VRAM (8-bit) |
|-------|------------|-------------|--------------|
| InternVL3-2B | 2B | ~6GB | ~4GB |
| Qwen2_5-VL-3B | 3B | ~8GB | ~5GB |
| **Qwen3-VL-4B** | 4B | ~10GB | ~6GB |
| Qwen2_5-VL-7B | 7B | ~16GB | ~10GB |
| InternVL3-8B | 8B | ~18GB | ~11GB |
| InternVL3-14B | 14B | ~30GB | ~18GB |

## Troubleshooting

**"transformers >= 4.57.0 required"**
```bash
pip install transformers>=4.57.0 --upgrade
```

**"No module named 'qwen_vl_utils'"**
```bash
pip install qwen_vl_utils
```

**"CUDA Out of Memory"**
- Enable `use_8bit`
- Use smaller model (2B/4B)
- Ensure `unload = True`

**"No masks generated"**
- Try more specific prompts
- Adjust `threshold` (try 0.3-0.7)

## Technical Details

### Raw Mask Probabilities
Sa2VA outputs raw sigmoid probabilities (0.0-1.0) instead of binary masks. The `threshold` parameter controls binarization.

### 8-bit Quantization
Quantizes language model backbone while skipping vision components (visual, grounding_encoder, text_hidden_fcs) to avoid errors.

### VITMatte (V2 Only)

VITMatte is a Vision Transformer-based alpha matting model that produces smooth, professional-quality masks.

**Why VITMatte produces better masks:**
- **Trimap-guided**: Creates a 3-zone map (definite foreground, uncertain region, definite background) from Sa2VA's rough mask
- **AI refinement**: Neural network predicts precise alpha values in uncertain regions (edges, hair, semi-transparent areas)
- **Gradient transitions**: Produces smooth 0-1 gradients instead of hard 0/1 boundaries
- **Detail preservation**: Captures fine structures like individual hair strands, fur texture, and glass transparency

**Processing pipeline:**
1. Generate trimap from sigmoid mask (erode/dilate)
2. VITMatte AI inference for alpha prediction in uncertain regions
3. Histogram remapping for contrast enhancement

**Trade-offs:**
- +2-5s processing time per mask
- +2GB VRAM usage
- Best for complex edges; V1 is faster for simple objects

## Links

- [Sa2VA Paper](https://arxiv.org/abs/2501.04001)
- [Sa2VA GitHub](https://github.com/bytedance/Sa2VA)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## License

MIT

## Credits

- Based on [ByteDance Sa2VA](https://github.com/bytedance/Sa2VA)
- Inspired by [ComfyUI-Sa2VA](https://github.com/adambarbato/ComfyUI-Sa2VA)
- VITMatte implementation adapted from [ComfyUI_LayerStyle_Advance](https://github.com/chflame163/ComfyUI_LayerStyle_Advance)
