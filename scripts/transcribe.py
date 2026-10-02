# -*- coding: utf-8 -*-
"""
Whisper-DML-AMD 核心识别脚本
用 DirectML (AMD/Intel/任意DX12显卡) 加速 Whisper 语音识别，输出 .srt + .txt

用法:
    python transcribe.py <视频或音频文件>
    python transcribe.py <目录>            # 递归处理目录下所有音视频
    python transcribe.py <文件> --device 0 # 指定GPU设备id

依赖: onnxruntime-directml, transformers, faster-whisper
"""
import argparse, os, re, subprocess, sys, time, wave, json
import numpy as np
import onnxruntime as ort
import mel as mel_module  # 内嵌的 mel 频谱计算

# ---------- 常量 ----------
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# Whisper 51865 词表特殊 token id（务必用这些值，写错一个就乱码）
SOT = 50258          # <|startoftranscript|>
LANG_ZH = 50260      # <|zh|>
TRANSCRIBE = 50359   # <|transcribe|>  ← 注意：不是 50259
NO_TS = 50363        # <|notimestamps|>
EOT = 50257          # <|endoftext|>

CHUNK_SEC = 30       # 分段长度（秒），whisper 标准 30s 窗口
MAX_TOKEN = 200      # 每段最多解码 token 数
SAMPLE_RATE = 16000

# ---------- 模型加载 ----------
def load_models(device_id=1):
    """加载 encoder + decoder（DML provider）"""
    opts = ort.SessionOptions()
    opts.enable_mem_pattern = False          # DML 必需
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # DML 必需
    prov = [("DmlExecutionProvider", {"device_id": device_id})]
    enc = ort.InferenceSession(os.path.join(MODEL_DIR, "encoder_model.onnx"),
                               sess_options=opts, providers=prov)
    dec = ort.InferenceSession(os.path.join(MODEL_DIR, "decoder_model.onnx"),
                               sess_options=opts, providers=prov)
    return enc, dec

def load_tokenizer():
    from transformers import WhisperTokenizer
    return WhisperTokenizer.from_pretrained(MODEL_DIR)

# ---------- 音频处理 ----------
def extract_audio(video_path, wav_path):
    """用 ffmpeg 抽 16kHz 单声道 wav"""
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-y", "-i", video_path, "-vn", "-ac", "1",
                    "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", wav_path],
                   capture_output=True, check=True)

def load_wav(wav_path):
    w = wave.open(wav_path)
    audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    w.close()
    return audio

# ---------- 识别 ----------
def transcribe_chunk(enc, dec, tok, audio_chunk):
    """识别一段音频（<=30s），返回文本"""
    # mel 频谱（精确 3000 帧）
    mel = mel_module.log_mel_spectrogram(audio_chunk)
    mel = mel_module.pad_or_trim(mel)  # [80, 3000]
    mel = mel[None].astype(np.float32)  # [1, 80, 3000]
    hs = enc.run(None, {"input_features": mel})[0]  # [1,1500,768]

    seq = [SOT, LANG_ZH, TRANSCRIBE, NO_TS]
    for _ in range(MAX_TOKEN):
        ids = np.array([seq], dtype=np.int64)
        logits = dec.run(None, {"input_ids": ids, "encoder_hidden_states": hs})[0]
        next_tok = int(np.argmax(logits[0, -1, :]))
        if next_tok == EOT:
            break
        seq.append(next_tok)
    return tok.decode(seq[4:], skip_special_tokens=True)  # 跳过4个特殊token

# ---------- 输出 ----------
def fmt_srt(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def split_sentences(text, max_len=40):
    """按标点粗切分字幕"""
    parts = re.findall(r".{1,%d}[。！？；，,.!?;]?|.+" % max_len, text)
    return [p.strip() for p in parts if p.strip()]

def write_outputs(base_path, chunks_text, chunk_starts, total_sec):
    """写 .srt 和 .txt"""
    # srt：每个 chunk 按句切分
    srt_lines, idx = [], 1
    for txt, st in zip(chunks_text, chunk_starts):
        for sent in split_sentences(txt):
            en = min(st + CHUNK_SEC, total_sec)
            srt_lines.append(f"{idx}\n{fmt_srt(st)} --> {fmt_srt(en)}\n{sent}\n")
            idx += 1
    open(base_path + ".srt", "w", encoding="utf-8").write("\n".join(srt_lines))

    # txt：合并成段落（160字/段）
    full = "".join(chunks_text)
    paras, cur = [], ""
    for ch in full:
        cur += ch
        if len(cur) >= 160 and ch in "。！？":
            paras.append(cur); cur = ""
    if cur:
        paras.append(cur)
    open(base_path + ".txt", "w", encoding="utf-8").write("\n\n".join(paras) + "\n")

# ---------- 主流程 ----------
def process_one(enc, dec, tok, input_path, device_id):
    """处理单个文件"""
    base = os.path.splitext(input_path)[0]
    if os.path.exists(base + ".srt") and os.path.getsize(base + ".srt") > 0:
        print(f"  [跳过] {os.path.basename(input_path)} 已有字幕")
        return

    # 抽音频
    wav = base + ".wav"
    if not os.path.exists(wav) or os.path.getsize(wav) < 1000:
        extract_audio(input_path, wav)
    audio = load_wav(wav)
    total_sec = len(audio) / SAMPLE_RATE
    print(f"  [{os.path.basename(input_path)}] {total_sec/60:.1f} 分钟", flush=True)

    t0 = time.time()
    chunks_text, chunk_starts = [], []
    n = int(np.ceil(total_sec / CHUNK_SEC))
    for i in range(n):
        chunk = audio[i * SAMPLE_RATE * CHUNK_SEC : (i + 1) * SAMPLE_RATE * CHUNK_SEC]
        if len(chunk) < SAMPLE_RATE:  # 不足1秒跳过
            break
        txt = transcribe_chunk(enc, dec, tok, chunk)
        chunks_text.append(txt)
        chunk_starts.append(i * CHUNK_SEC)

    write_outputs(base, chunks_text, chunk_starts, total_sec)
    el = time.time() - t0
    print(f"  ✓ 完成，{el:.0f}s（{total_sec/el:.1f} 倍实时）", flush=True)

    # 清理临时 wav（可选，注释掉则保留）
    # os.remove(wav)

def main():
    parser = argparse.ArgumentParser(description="AMD DirectML Whisper 语音识别")
    parser.add_argument("input", help="视频/音频文件 或 目录")
    parser.add_argument("--device", type=int, default=1, help="GPU 设备id（默认1，通常0是核显/虚拟卡）")
    args = parser.parse_args()

    print("[加载] DirectML 模型 ...")
    enc, dec = load_models(args.device)
    tok = load_tokenizer()
    

    # 收集待处理文件
    VIDEO_EXT = (".mp4", ".mkv", ".flv", ".avi", ".mov", ".wav", ".mp3", ".m4a")
    if os.path.isdir(args.input):
        files = []
        for root, _, fs in os.walk(args.input):
            for f in fs:
                if f.lower().endswith(VIDEO_EXT):
                    files.append(os.path.join(root, f))
        files.sort()
    else:
        files = [args.input]

    print(f"共 {len(files)} 个文件")
    for fp in files:
        try:
            process_one(enc, dec, tok, fp, args.device)
        except Exception as e:
            print(f"  [错误] {fp}: {e}", flush=True)
    print("\n[完成]")

if __name__ == "__main__":
    main()
