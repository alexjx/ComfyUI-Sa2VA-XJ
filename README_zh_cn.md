# ComfyUI-Sa2VA-XJ

中文文档 | [English](README.md)

[ByteDance Sa2VA](https://github.com/bytedance/Sa2VA) 在 ComfyUI 中的简单实现。

## 功能特性

- ✅ **三个节点**: 图像 V1、图像 V2 和视频处理
- ✅ **符合 ComfyUI 规范的模型路径**: 支持手动下载,优先使用本地模型
- ✅ **VITMatte 后处理 (V2)**: AI 驱动的 alpha 抠图
- ✅ **可配置的掩码阈值**: 控制掩码质量 (0.0-1.0, 步长 0.05)
- ✅ **形态学操作 (V1)**: 开运算、闭运算、腐蚀、膨胀
- ✅ **8位量化**: 节省显存
- ✅ **Flash attention**: 更快的推理速度
- ✅ **模型卸载**: 推理后释放显存

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/alexjx/ComfyUI-Sa2VA-XJ.git
cd ComfyUI-Sa2VA-XJ
pip install -r requirements.txt
```

**可选依赖:**
```bash
pip install bitsandbytes              # 用于8位量化
pip install flash-attn --no-build-isolation  # 用于 flash attention
pip install opencv-python              # 用于形态学操作
```

安装后需要重启 ComfyUI。

## 模型安装

您可以**手动**下载模型(推荐)或让节点在首次使用时从 HuggingFace 自动下载。

### 方案 1: 手动下载(推荐)

手动下载可以让您控制模型位置,支持离线使用,并避免重复下载。

**目录结构:**
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
│   └── (或不使用 ByteDance/ 前缀)
└── vitmatte/
    └── hustvl/
        └── vitmatte-small-composition-1k/
```

**方法 1: 使用 huggingface-cli (推荐)**

```bash
# 安装 HuggingFace CLI
pip install -U huggingface_hub

# 下载 Sa2VA 模型 (示例: Qwen3-VL-4B)
huggingface-cli download ByteDance/Sa2VA-Qwen3-VL-4B \
  --local-dir ComfyUI/models/sa2va/ByteDance/Sa2VA-Qwen3-VL-4B

# 下载 VITMatte 模型 (用于 V2 节点)
huggingface-cli download hustvl/vitmatte-small-composition-1k \
  --local-dir ComfyUI/models/vitmatte/hustvl/vitmatte-small-composition-1k
```

**方法 2: 使用 git-lfs**

```bash
# 安装 git-lfs
git lfs install

# 下载 Sa2VA 模型
cd ComfyUI/models/sa2va/ByteDance
git clone https://huggingface.co/ByteDance/Sa2VA-Qwen3-VL-4B

# 下载 VITMatte 模型
cd ComfyUI/models/vitmatte/hustvl
git clone https://huggingface.co/hustvl/vitmatte-small-composition-1k
```

**替代目录结构:**

您也可以不使用组织名称前缀直接下载:

```bash
# 直接下载到模型名称文件夹
huggingface-cli download ByteDance/Sa2VA-Qwen3-VL-4B \
  --local-dir ComfyUI/models/sa2va/Sa2VA-Qwen3-VL-4B
```

两种目录结构都支持:
- `models/sa2va/ByteDance/Sa2VA-Qwen3-VL-4B/` ✅
- `models/sa2va/Sa2VA-Qwen3-VL-4B/` ✅

### 方案 2: 自动下载

如果本地未找到模型,首次使用时将自动从 HuggingFace 下载到 `~/.cache/huggingface/hub/`。

**注意:** 这可能需要一些时间,取决于您的网络连接,并会使用 HuggingFace 的缓存目录。

### 模型大小

请相应规划您的存储空间:

| 模型                          | 下载大小 | 磁盘空间(已安装) | 显存 (fp16) | 显存 (8-bit) |
| ----------------------------- | -------- | ---------------- | ----------- | ------------ |
| **Sa2VA 模型**                |
| InternVL3-2B                  | ~4GB     | ~4.5GB           | ~6GB        | ~4GB         |
| Qwen2_5-VL-3B                 | ~6GB     | ~6.5GB           | ~8GB        | ~5GB         |
| **Qwen3-VL-4B** (默认)        | **~8GB** | **~9GB**         | **~10GB**   | **~6GB**     |
| Qwen2_5-VL-7B                 | ~14GB    | ~15GB            | ~16GB       | ~10GB        |
| InternVL3-8B                  | ~16GB    | ~17GB            | ~18GB       | ~11GB        |
| InternVL3-14B                 | ~28GB    | ~30GB            | ~30GB       | ~18GB        |
| **VITMatte 模型**             |
| vitmatte-small-composition-1k | ~300MB   | ~350MB           | +2GB        | +2GB         |

**存储建议:**
- 从 **Qwen3-VL-4B** (默认)开始 - 质量和速度的良好平衡
- 使用 **8位量化** 来减少显存使用
- VITMatte 是可选的(仅 V2 节点)用于增强边缘质量

## 系统要求

- Python 3.8+
- PyTorch 2.0+
- transformers >= 4.57.0
- CUDA 11.8+ (GPU)
- 显存: 8GB+ (2B-4B 模型), 16GB+ (7B-8B 模型), 24GB+ (14B 模型)

## 节点说明

### Sa2VA 图像分割 (V1)

**输入参数:**
- `model_name`: 模型选择(默认: Qwen3-VL-4B)
- `image`: 输入图像
- `segmentation_prompt`: 文本描述
- `threshold`: 二值化阈值 (0.0-1.0, 默认: 0.5)
- `use_8bit`: 8位量化 (默认: True)
- `use_flash_attn`: Flash attention (默认: True)
- `unload`: 推理后卸载模型 (默认: True)
- `morph`: 形态学操作 (none/opening/closing/erode/dilate)
- `erode_kernel`, `dilate_kernel`, `iterations`: 形态学参数

**输出:**
- `text_output`: 文本描述
- `masks`: 分割掩码

### Sa2VA 图像分割 V2

基于 VITMatte 的后处理,生成平滑的 alpha 遮罩。

**输入参数:**
- `model_name`: 模型选择(默认: Qwen3-VL-4B)
- `image`: 输入图像
- `segmentation_prompt`: 文本描述
- `threshold`: 二值化阈值 (0.0-1.0, 默认: 0.5)
- `use_8bit`: 8位量化 (默认: True)
- `use_flash_attn`: Flash attention (默认: True)
- `unload`: 推理后卸载模型 (默认: True)
- `process_detail`: 启用 VITMatte (默认: True)
- `detail_erode`: 三值图腐蚀大小 (1-255, 默认: 6)
- `detail_dilate`: 三值图膨胀大小 (1-255, 默认: 6)
- `black_point`: 直方图黑点 (0.01-0.98, 默认: 0.15)
- `white_point`: 直方图白点 (0.02-0.99, 默认: 0.99)
- `max_megapixels`: 最大分辨率 (0.5-10.0, 默认: 2.0)

**输出:**
- `text_output`: 文本描述
- `masks`: Alpha 遮罩

**V1 与 V2 对比:**
- V1: 快速,形态学操作,适合实心物体
- V2: 较慢,VITMatte 精细化,适合头发/毛发/玻璃/复杂边缘

### Sa2VA 视频分割

处理视频帧或图像批次。

**输入参数:**
- `model_name`: 模型选择
- `images`: 输入帧(批次)
- `segmentation_prompt`: 文本描述
- `threshold`: 二值化阈值 (0.0-1.0, 默认: 0.7)
- `use_8bit`, `use_flash_attn`, `unload`: 与 V1 相同
- `morph`, `erode_kernel`, `dilate_kernel`, `iterations`: 形态学参数

**输出:**
- `text_output`: 视频描述
- `masks`: 所有帧的分割掩码

## 支持的模型

所有模型均来自 ByteDance 的 Sa2VA 系列:

| 模型            | 参数量 | 说明                        |
| --------------- | ------ | --------------------------- |
| InternVL3-2B    | 2B     | 最小,最快                   |
| Qwen2_5-VL-3B   | 3B     | 适合低显存                  |
| **Qwen3-VL-4B** | 4B     | **默认 - 最佳平衡**         |
| Qwen2_5-VL-7B   | 7B     | 更高质量                    |
| InternVL3-8B    | 8B     | 高级功能                    |
| InternVL3-14B   | 14B    | 最佳质量,需要 24GB+ 显存    |

请参阅[模型安装](#模型安装)部分了解下载说明和显存要求。

## 故障排除

**"transformers >= 4.57.0 required"**
```bash
pip install transformers>=4.57.0 --upgrade
```

**"No module named 'qwen_vl_utils'"**
```bash
pip install qwen_vl_utils
```

**模型下载**

节点会记录从哪里加载模型:
- `Found local Sa2VA model at: /path/to/model` - 使用本地模型 ✅
- `Local model not found. Will download from HuggingFace: ...` - 从 HF 自动下载

验证模型安装:
```bash
# 检查 Sa2VA 模型是否存在
ls -la ComfyUI/models/sa2va/

# 检查 VITMatte 模型是否存在
ls -la ComfyUI/models/vitmatte/

# 每个模型目录应包含 config.json
ls ComfyUI/models/sa2va/ByteDance/Sa2VA-Qwen3-VL-4B/config.json
```

**首次加载缓慢**
- 首次运行可能会从 HuggingFace 下载模型(可能需要 5-30 分钟)
- 查看控制台日志以了解下载进度
- 考虑手动下载(参见[模型安装](#模型安装))

**"CUDA Out of Memory"**
- 启用 `use_8bit`
- 使用更小的模型 (2B/4B)
- 确保 `unload = True`
- 降低 VITMatte 的 `max_megapixels` (仅 V2)

**"No masks generated"**
- 尝试更具体的提示词
- 调整 `threshold` (尝试 0.3-0.7)

## 技术细节

### 模型加载行为

节点遵循 ComfyUI 的标准模型加载模式:

1. **优先检查本地模型**: 在 `ComfyUI/models/sa2va/` 和 `ComfyUI/models/vitmatte/` 中查找
2. **回退到 HuggingFace**: 如果本地未找到,自动下载到 `~/.cache/huggingface/hub/`
3. **支持两种目录结构**:
   - 带组织前缀: `models/sa2va/ByteDance/Sa2VA-Qwen3-VL-4B/`
   - 不带组织前缀: `models/sa2va/Sa2VA-Qwen3-VL-4B/`

**优势:**
- ✅ 控制模型存储位置
- ✅ 手动下载后支持离线使用
- ✅ 无重复下载
- ✅ 向后兼容自动下载

### 原始掩码概率

Sa2VA 输出原始 sigmoid 概率 (0.0-1.0) 而不是二值掩码。`threshold` 参数控制二值化。

### 8位量化

量化语言模型骨干网络,同时跳过视觉组件(visual、grounding_encoder、text_hidden_fcs)以避免错误。

### VITMatte (仅 V2)

VITMatte 是一个基于 Vision Transformer 的 alpha 抠图模型,可生成平滑、专业质量的掩码。

**为什么 VITMatte 能生成更好的掩码:**
- **三值图引导**: 从 Sa2VA 的粗略掩码创建 3 区域图(确定前景、不确定区域、确定背景)
- **AI 精细化**: 神经网络在不确定区域(边缘、头发、半透明区域)预测精确的 alpha 值
- **渐变过渡**: 生成平滑的 0-1 渐变而不是硬性的 0/1 边界
- **细节保留**: 捕捉精细结构,如单根头发丝、毛发纹理和玻璃透明度

**处理流程:**
1. 从 sigmoid 掩码生成三值图(腐蚀/膨胀)
2. VITMatte AI 推理在不确定区域预测 alpha
3. 直方图重映射以增强对比度

**权衡:**
- 每个掩码增加 2-5 秒处理时间
- 增加 2GB 显存使用
- 最适合复杂边缘;V1 对简单物体更快

## 相关链接

- [Sa2VA 论文](https://arxiv.org/abs/2501.04001)
- [Sa2VA GitHub](https://github.com/bytedance/Sa2VA)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## 许可证

MIT

## 致谢

- 基于 [ByteDance Sa2VA](https://github.com/bytedance/Sa2VA)
- 灵感来自 [ComfyUI-Sa2VA](https://github.com/adambarbato/ComfyUI-Sa2VA)
- VITMatte 实现改编自 [ComfyUI_LayerStyle_Advance](https://github.com/chflame163/ComfyUI_LayerStyle_Advance)
