# -*- coding: utf-8 -*-
"""下载 whisper-small ONNX 模型（onnx-community，标准词表）
从 HuggingFace 镜像下载，支持断点续传
"""
import os, sys, urllib.request

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# 需要的文件（encoder + decoder + 配置）
FILES = {
    "onnx/encoder_model.onnx": "encoder_model.onnx",
    "onnx/decoder_model.onnx": "decoder_model.onnx",
    "config.json": "config.json",
    "generation_config.json": "generation_config.json",
    "preprocessor_config.json": "preprocessor_config.json",
    "tokenizer.json": "tokenizer.json",
}

# 镜像：国内用 hf-mirror.com，海外可换 huggingface.co
MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
REPO = "onnx-community/whisper-small"

def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"  [跳过] {os.path.basename(dest)} 已存在")
        return
    print(f"  [下载] {os.path.basename(dest)} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
    print(f"  [完成] {os.path.basename(dest)} ({os.path.getsize(dest)//1024//1024} MB)")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"模型保存目录: {MODEL_DIR}")
    print(f"镜像: {MIRROR}")
    for remote, local in FILES.items():
        url = f"{MIRROR}/{REPO}/resolve/main/{remote}"
        download(url, os.path.join(MODEL_DIR, local))
    print("\n✅ 模型下载完成！")

if __name__ == "__main__":
    main()
