# Whisper-DML-AMD 🎙️

**Windows + AMD 显卡（Radeon）跑 Whisper 语音识别的完整方案**，无需 CUDA、无需 ROCm、无需付费会员。

用**混合架构**把 AMD 显卡和 CPU 都用上：GPU 跑 encoder（音频编码），CPU 跑 decoder（文字生成），生成 `.srt` 字幕 + `.txt` 讲稿。

> 项目源自一次真实需求：给 **28.7 小时**网课视频批量生成字幕和文字讲义。
> 实测 **AMD RX 6750 XT** 达到 **6.1 倍实时**（纯 CPU 仅 1.7 倍，快 3.6 倍）。

---

## ✨ 特性

- ✅ **混合架构**：GPU encoder + CPU decoder，速度 6.1 倍实时
- ✅ 纯 AMD 显卡加速，Windows 原生，**无需 CUDA / ROCm / WSL**
- ✅ 中文识别准确（whisper-small，支持 99 种语言）
- ✅ 输出 `.srt` 字幕（播放器自动加载）+ `.txt` 纯文本讲稿
- ✅ 30 秒分段处理，长视频不爆显存
- ✅ 断点续传，中断后自动跳过已完成部分

## 📊 性能实测

环境：AMD RX 6750 XT (12GB) + 16 核 CPU，3 分钟中文音频

| 方案 | 速度 | 28.7h 全量耗时 | 说明 |
|---|---|---|---|
| 纯 CPU (faster-whisper) | 1.7 倍 | ~17 小时 | 基准 |
| DML 全量重算 | 3.6 倍 | ~8 小时 | decoder 每步重算整个序列 |
| **混合架构（推荐）** | **6.1 倍** | **~4.7 小时** | GPU encoder + CPU 增量 decoder |

分项耗时（30 秒音频）：
- encoder（GPU）：0.3 秒 → **100 倍实时**
- decoder（CPU 增量 KV cache）：约 6 秒

## 🚀 快速开始

### 1. 安装依赖

```bash
# 必须独立虚拟环境！onnxruntime-directml 与标准 onnxruntime 包名冲突
python -m venv dml_env
dml_env\Scripts\activate

pip install onnxruntime-directml transformers imageio-ffmpeg numpy
```

### 2. 下载模型

```bash
python scripts/download_model.py
```

自动从 HuggingFace 镜像下载 onnx-community/whisper-small（约 1.5GB）。

### 3. 运行

```bash
# 单个视频
python scripts/transcribe_hybrid.py 你的视频.mp4

# 整个目录（递归，输出到视频旁边）
python scripts/transcribe_hybrid.py "D:\课程视频" --outdir "D:\课程视频"
```

每个视频生成同名的 `.srt`（字幕）和 `.txt`（讲稿）。

### 4. 繁转简（中文用户必做）

Whisper 的中文识别**默认偏向输出繁体**（「各位同學」「高等數學」），需要后处理：

```bash
pip install opencc-python-reimplemented
python scripts/simplify_subs.py "D:\课程视频"
```

英文内容不受影响，只转汉字。

## 📖 为什么是混合架构？

这是本项目最核心的发现。Whisper 由两部分组成，它们的硬件偏好完全不同：

```
音频 ──► encoder（Transformer 编码）
            │  并行度高、计算密集 → GPU 快（100 倍实时）
            ▼
         hidden states
            │
            ▼
         decoder（逐 token 自回归生成）
            │  串行、每步只出 1 个 token
            │  需要动态增长的 KV cache
            ▼
         文本 token
```

**关键问题：DirectML 不支持动态 KV cache。**

decoder 增量解码时，KV cache 每步增长 1（形状 `[1,12,0,64]` → `[1,12,1,64]` → ...）。
- DML 每次遇到新形状都要重新编译 shader，导致**每步 111ms**，比 CPU 还慢，且第二步就直接崩溃（`ScatterND`/`Gather` 动态索引不支持）
- CPU 上用 onnxruntime 标准版做增量解码反而更快更稳

**所以最优解是分工**：GPU 干它擅长的 encoder，CPU 干它擅长的增量 decoder。

> 顺带开发了纯 DML 版本（`transcribe.py`，全量重算 3.6 倍），作为备选。它是「零 CPU 解码」场景的可行方案。

## 🔑 踩坑记录（能省你几小时）

| # | 坑 | 结论 |
|---|---|---|
| 1 | `torch-directml` 装不上 | 只支持 Python≤3.12 且官方已停更，**别用** |
| 2 | `onnxruntime-directml` 和 `onnxruntime` 冲突 | 两者模块同名，必须**独立 venv** |
| 3 | sherpa-onnx 的 medium int8 模型 DML 崩 | 换 **small fp32** 模型 |
| 4 | sherpa-onnx PyPI 版 DML 被禁用 | wheel 在 Linux 编译，Windows 上硬编码回退 CPU，**别用 sherpa 的 provider** |
| 5 | 手写 mel 预处理出错 | 直接复用/内嵌标准 Slaney mel 实现，注意**精确截到 3000 帧**（多一帧就崩） |
| 6 | sherpa tokens.txt 是 base64 且无特殊 token | 用 **onnx-community 模型**（标准 51865 词表） |
| 7 | token id 写错 | `transcribe=50359`（**不是 50259**！）、`SOT=50258`、`zh=50260`、`notimestamps=50363`、`EOT=50257` |
| 8 | decoder 选型 | `decoder_model.onnx`（全量重算，简单）/ `decoder_model_merged.onnx`（支持 KV cache，混合架构用它） |
| 9 | 增量解码乱码 | 第一步 `use_cache_branch=False` **只喂 SOT**；语言 token（`<|zh|>`）由模型**自动生成**，不能预置；之后 `use_cache_branch=True` 只喂新 token，且**只更新 decoder cache**（encoder cache 复用） |
| 10 | 中文输出是繁体 | Whisper 中文识别的默认偏好，用 `simplify_subs.py`（OpenCC t2s）后处理转简体 |

### 特殊 token id 速查（whisper 51865 词表）

```python
SOT         = 50258  # <|startoftranscript|>
LANG_ZH     = 50260  # <|zh|>
TRANSCRIBE  = 50359  # <|transcribe|>  ← 注意不是 50259
NO_TS       = 50363  # <|notimestamps|>
EOT         = 50257  # <|endoftext|>
```

### GPU 使用率为什么只有 30%？

因为瓶颈在 CPU decoder（逐 token 串行）。GPU 大部分时间在等 CPU 生成下一个 token。
混合架构下 GPU 只负责 encoder，利用率看起来不高，但**整体吞吐是最优的**。

## 🛠️ 文件说明

```
whisper-dml-amd/
├── scripts/
│   ├── transcribe_hybrid.py   # ⭐ 推荐：混合架构（GPU encoder + CPU decoder）
│   ├── transcribe.py          # 备选：纯 DML 全量重算（无 CPU 解码）
│   ├── mel.py                 # log-mel 频谱（独立实现，无重依赖）
│   ├── download_model.py      # 模型下载（hf-mirror 镜像）
│   ├── batch_parallel.py      # 多进程并行批处理
│   ├── worker_driver.py       # 并行 worker
│   ├── simplify_subs.py       # 繁转简后处理（中文用户必做）
│   └── organize_subs.py       # 字幕归位 + 清理临时文件
├── README.md
└── LICENSE
```

## 📄 License

MIT

## 🙏 致谢

- [onnx-community/whisper-small](https://huggingface.co/onnx-community/whisper-small) - 标准词表 Whisper ONNX 模型
- [microsoft/onnxruntime-directml](https://pypi.org/project/onnxruntime-directml/) - DirectML 执行后端
- [openai/whisper](https://github.com/openai/whisper) / [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) - 参考实现与 mel 预处理
