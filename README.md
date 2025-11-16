# ComfyUI-Sa2VA-XJ

Simple implementation of [ByteDance Sa2VA](https://github.com/bytedance/Sa2VA) nodes for ComfyUI.

## Features

- ✅ **Three nodes**: Image V1, Image V2, and Video processing
- ✅ **ComfyUI-compliant model paths**: Manual download support, local model priority
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

## Model Installation

You can download models **manually** (recommended) or let the node auto-download from HuggingFace on first use.

### Option 1: Manual Download (Recommended)

Manual download gives you control over model locations, enables offline use, and avoids duplicate downloads.

**Directory Structure:**
```
ComfyUI/models/
├── sa2va/
│   ├── ByteDance/
│   │   ├── Sa2VA-Qwen3-VL-4B/
│   │   ├── Sa2VA-InternVL3-2B/
│   │   ├── Sa2VA-Qwen2_5-VL-3B/
│   │   ├── Sa2VA-Qwen2_5-VL-7B/
│   │   ├── Sa2VA-InternVL3-8B/
│   │   └── Sa2VA-InternVL3-14B/
│   └── (or without ByteDance/ prefix)
└── vitmatte/
    └── hustvl/
        └── vitmatte-small-composition-1k/
```

**Method 1: Using huggingface-cli (Recommended)**

```bash
# Install HuggingFace CLI
pip install -U huggingface_hub

# Download Sa2VA model (example: Qwen3-VL-4B)
huggingface-cli download ByteDance/Sa2VA-Qwen3-VL-4B \
  --local-dir ComfyUI/models/sa2va/ByteDance/Sa2VA-Qwen3-VL-4B

# Download VITMatte model (for V2 node)
huggingface-cli download hustvl/vitmatte-small-composition-1k \
  --local-dir ComfyUI/models/vitmatte/hustvl/vitmatte-small-composition-1k
```

**Method 2: Using git-lfs**

```bash
# Install git-lfs
git lfs install

# Download Sa2VA model
cd ComfyUI/models/sa2va/ByteDance
git clone https://huggingface.co/ByteDance/Sa2VA-Qwen3-VL-4B

# Download VITMatte model
cd ComfyUI/models/vitmatte/hustvl
git clone https://huggingface.co/hustvl/vitmatte-small-composition-1k
```

**Alternative Directory Structure:**

You can also download without the organization prefix:

```bash
# Download directly to model name folder
huggingface-cli download ByteDance/Sa2VA-Qwen3-VL-4B \
  --local-dir ComfyUI/models/sa2va/Sa2VA-Qwen3-VL-4B
```

Both structures are supported:
- `models/sa2va/ByteDance/Sa2VA-Qwen3-VL-4B/` ✅
- `models/sa2va/Sa2VA-Qwen3-VL-4B/` ✅

### Option 2: Automatic Download

Models will auto-download from HuggingFace to `~/.cache/huggingface/hub/` on first use if not found locally.

**Note:** This may take time depending on your internet connection and will use HuggingFace's cache directory.

### Model Sizes

Plan your storage accordingly:

| Model                         | Download Size | Disk Size (Installed) | VRAM (fp16) | VRAM (8-bit) |
| ----------------------------- | ------------- | --------------------- | ----------- | ------------ |
| **Sa2VA Models**              |
| InternVL3-2B                  | ~4GB          | ~4.5GB                | ~6GB        | ~4GB         |
| Qwen2_5-VL-3B                 | ~6GB          | ~6.5GB                | ~8GB        | ~5GB         |
| **Qwen3-VL-4B** (default)     | **~8GB**      | **~9GB**              | **~10GB**   | **~6GB**     |
| Qwen2_5-VL-7B                 | ~14GB         | ~15GB                 | ~16GB       | ~10GB        |
| InternVL3-8B                  | ~16GB         | ~17GB                 | ~18GB       | ~11GB        |
| InternVL3-14B                 | ~28GB         | ~30GB                 | ~30GB       | ~18GB        |
| **VITMatte Model**            |
| vitmatte-small-composition-1k | ~300MB        | ~350MB                | +2GB        | +2GB         |

**Storage Tips:**
- Start with **Qwen3-VL-4B** (default) - good balance of quality and speed
- Use **8-bit quantization** to reduce VRAM usage
- VITMatte is optional (V2 node only) for enhanced edge quality

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

All models are from ByteDance's Sa2VA family:

| Model           | Parameters | Notes                             |
| --------------- | ---------- | --------------------------------- |
| InternVL3-2B    | 2B         | Smallest, fastest                 |
| Qwen2_5-VL-3B   | 3B         | Good for low VRAM                 |
| **Qwen3-VL-4B** | 4B         | **Default - Best balance**        |
| Qwen2_5-VL-7B   | 7B         | Higher quality                    |
| InternVL3-8B    | 8B         | Advanced features                 |
| InternVL3-14B   | 14B        | Best quality, requires 24GB+ VRAM |

See [Model Installation](#model-installation) section for download instructions and VRAM requirements.

## Troubleshooting

**"transformers >= 4.57.0 required"**
```bash
pip install transformers>=4.57.0 --upgrade
```

**"No module named 'qwen_vl_utils'"**
```bash
pip install qwen_vl_utils
```

**Model Downloads**

The node will log where it's loading models from:
- `Found local Sa2VA model at: /path/to/model` - Using local model ✅
- `Local model not found. Will download from HuggingFace: ...` - Auto-downloading from HF

To verify model installation:
```bash
# Check if Sa2VA models exist
ls -la ComfyUI/models/sa2va/

# Check if VITMatte model exists
ls -la ComfyUI/models/vitmatte/

# Each model directory should contain config.json
ls ComfyUI/models/sa2va/ByteDance/Sa2VA-Qwen3-VL-4B/config.json
```

**Slow First Load**
- First run may download models from HuggingFace (can take 5-30 minutes)
- Check console logs to see download progress
- Consider manual download (see [Model Installation](#model-installation))

**"CUDA Out of Memory"**
- Enable `use_8bit`
- Use smaller model (2B/4B)
- Ensure `unload = True`
- Lower VITMatte `max_megapixels` (V2 only)

**"No masks generated"**
- Try more specific prompts
- Adjust `threshold` (try 0.3-0.7)

## Technical Details

### Model Loading Behavior

The nodes follow ComfyUI's standard model loading pattern:

1. **Check local models first**: Looks in `ComfyUI/models/sa2va/` and `ComfyUI/models/vitmatte/`
2. **Fallback to HuggingFace**: Auto-downloads to `~/.cache/huggingface/hub/` if not found locally
3. **Supports both directory structures**:
   - With org prefix: `models/sa2va/ByteDance/Sa2VA-Qwen3-VL-4B/`
   - Without org prefix: `models/sa2va/Sa2VA-Qwen3-VL-4B/`

**Benefits:**
- ✅ Control over model storage locations
- ✅ Offline use after manual download
- ✅ No duplicate downloads
- ✅ Backward compatible with auto-download

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
