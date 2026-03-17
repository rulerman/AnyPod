[English](README.md) | [简体中文](README_zh.md)

# AnyPod

<p align="center">
    <img src="assets/anypod.png" alt="AnyPod" width="330">
  </p>

  <br>

AnyPod 是一个开源的自动化播客生成工具，基于 MOSS-TTSD 等开源 TTS 模型驱动。它可以将任意文本输入（TXT/PDF）转化为多集高质量播客节目。该工具通过 LLM Agent 自动分析输入文本、规划播客内容、生成播客剧本，并通过 TTS 合成为语音。支持自定义音色克隆、编辑节目设定和剧本，以及中英文双语的输入和输出。

## 安装

本工具支持四种 TTS 后端：

- **MOSS-TTSD（8B）**：生成效果最佳（推荐）。
- **MOSS-TTS（8B）**：单人模式生成效果最佳。
- **VibeVoice（1.5B）**：轻量级选择，适合大多数个人设备。
- **MOSS-TTS API**：环境配置最简单，但仅支持单人模式。

### 配置 TTS 环境

**如果你使用 MOSS-TTS / MOSS-TTSD 作为 TTS 后端：**

```bash
# 创建 MOSS-TTS / MOSS-TTSD 环境
conda create -n anypod_moss_tts python=3.11 -y
conda activate anypod_moss_tts
pip install -r requirements_moss_tts.txt
pip install flash-attn  # 安装 FlashAttention（可选）

# 下载模型权重
huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer \
  --local-dir model/MOSS-Audio-Tokenizer \
  --local-dir-use-symlinks False

huggingface-cli download OpenMOSS-Team/MOSS-TTSD-v1.0 \
  --local-dir model/MOSS-TTSD-v1.0 \
  --local-dir-use-symlinks False

huggingface-cli download OpenMOSS-Team/MOSS-TTS \
  --local-dir model/MOSS-TTS \
  --local-dir-use-symlinks False
```

**如果你使用 VibeVoice 作为 TTS 后端：**

```bash
# 创建 VibeVoice 环境
conda create -n anypod_vibevoice python=3.11 -y
conda activate anypod_vibevoice
pip install -r requirements_vibevoice.txt

# 下载模型权重
huggingface-cli download microsoft/VibeVoice-1.5B \
  --local-dir model/VibeVoice-1.5B \
  --local-dir-use-symlinks False

python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B')
tokenizer.save_pretrained('model/Qwen2.5-1.5B-tokenizer')
print('Done')
"
```

**如果你使用 MOSS-TTS API 作为 TTS 后端：**

无需配置额外的 TTS 环境。只需在 `config/llm_api_config.json` 的 `moss_tts_api` 部分填入你的 API Key 和 Voice ID 即可。

### 配置基础环境

```bash
conda create -n anypod python=3.11 -y
conda activate anypod
pip install -r requirements.txt
export ANYPOD_CONDA_HOME=YOUR_CONDA_PATH  # 例如：~/miniconda3
```

## LLM Agent 配置

在 `config/llm_api_config.json` 中配置 AnyPod 使用的 LLM Agent。每个 Agent 需要以下必填字段：

```json
{
  "base_url": "",
  "model": "",
  "api_key": ""
}
```

共有三个 Agent，分别适合不同类型的模型：

- **understanding_agent** — 推荐使用轻量级模型以获得更高性价比（如 Qwen3-Flash）。
- **plan_agent** — 推荐使用具有较强推理能力的模型（如 GPT-5.4 Thinking）。
- **writing_agent** — 推荐使用写作能力较强的模型（如 Gemini 3 Flash/Pro）。

## 使用方式

### 通过 Gradio Web UI 运行

```bash
python gradio_main.py \
  --server_name 127.0.0.1 \
  --server_port 7860
```

### 通过命令行运行

```bash
python main.py \
  --config_json config/anypod_config.json
```

请编辑 `config/anypod_config.json` 来设置输入参数。

## Coming Soon

- 支持接入更多 TTS 模型。
- 支持更多语言和更多说话人。
- 支持更多类型、更多模态的输入（如图片型 PDF）。
- Windows / macOS / Android 应用。

## Contributing

欢迎贡献！

1. Fork 本仓库
2. 创建功能分支
3. 进行修改
4. 提交 Pull Request

## License

本项目基于 [MIT License](LICENSE) 开源.
