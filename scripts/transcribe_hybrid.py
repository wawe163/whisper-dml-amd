# -*- coding: utf-8 -*-
"""
混合架构识别核心：DML encoder(GPU) + CPU 增量 decoder(KV cache)
这是最优解：6.1 倍实时（DML 全量重算仅 3.6 倍，纯 CPU 1.7 倍）

原理：
  - encoder 计算量大、并行度高 → 交给 GPU (DML)，100 倍实时
  - decoder 是逐 token 串行 + 动态 KV cache → DML 不支持动态形状，交给 CPU 增量解码
  - 语言 token (<|zh|>) 由模型自动生成，不预置
"""
import argparse, os, re, subprocess, time, wave
import numpy as np
import onnxruntime as ort
import mel as mel_module

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
SOT = 50258          # <|startoftranscript|>
EOT = 50257          # <|endoftext|>
CHUNK_SEC = 30
SAMPLE_RATE = 16000
MAX_TOKEN = 300      # 每段最多解码 token

def load_encoder(device_id=1):
    """encoder 用 DML(GPU)"""
    opts = ort.SessionOptions()
    opts.enable_mem_pattern = False
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(os.path.join(MODEL_DIR, "encoder_model.onnx"),
                                sess_options=opts,
                                providers=[("DmlExecutionProvider", {"device_id": device_id})])

def load_decoder():
    """decoder 用 CPU（增量 KV cache）"""
    return ort.InferenceSession(os.path.join(MODEL_DIR, "decoder_model_merged.onnx"),
                                providers=["CPUExecutionProvider"])

def load_tokenizer():
    from transformers import WhisperTokenizer
    return WhisperTokenizer.from_pretrained(MODEL_DIR)

def transcribe_chunk(enc, dec, tok, audio_chunk):
    """识别一段音频（<=30s），返回文本（混合架构）"""
    mel = mel_module.log_mel_spectrogram(audio_chunk)
    mel = mel_module.pad_or_trim(mel)
    mel = mel[None].astype(np.float32)
    hs = enc.run(None, {"input_features": mel})[0]  # GPU encoder

    # 初始化空 past_key_values（48个）
    past = {}
    for l in range(12):
        for modu in ("decoder", "encoder"):
            for kv in ("key", "value"):
                past[f"past_key_values.{l}.{modu}.{kv}"] = np.zeros((1, 12, 0, 64), dtype=np.float32)

    # 第一步：use_cache_branch=False，只喂 SOT
    out = dec.run(None, dict(input_ids=np.array([[SOT]], dtype=np.int64),
                             encoder_hidden_states=hs,
                             use_cache_branch=[False], **past))
    logits = out[0]; present = out[1:]
    for j, key in enumerate(past):
        past[key] = present[j]

    next_tok = int(np.argmax(logits[0, -1, :]))
    gen = [SOT, next_tok]

    # 后续：use_cache_branch=True，只喂最后 token，只更新 decoder cache
    while next_tok != EOT and len(gen) < MAX_TOKEN:
        out = dec.run(None, dict(input_ids=np.array([[next_tok]], dtype=np.int64),
                                 encoder_hidden_states=hs,
                                 use_cache_branch=[True], **past))
        logits = out[0]; present = out[1:]
        for j, key in enumerate(past):
            if "decoder" in key:
                past[key] = present[j]
        next_tok = int(np.argmax(logits[0, -1, :]))
        gen.append(next_tok)

    return tok.decode(gen[1:], skip_special_tokens=True)

# ---------- 输出（复用）----------
def fmt_srt(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def split_sentences(text, max_len=40):
    parts = re.findall(r".{1,%d}[。！？；，,.!?;]?|.+" % max_len, text)
    return [p.strip() for p in parts if p.strip()]

def extract_audio(video_path, wav_path):
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-y", "-i", video_path, "-vn", "-ac", "1",
                    "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", wav_path],
                   capture_output=True, check=True)

def load_wav(wav_path):
    w = wave.open(wav_path)
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    w.close()
    return a

def write_outputs(base_path, chunks_text, chunk_starts, total_sec):
    srt_lines, idx = [], 1
    for txt, st in zip(chunks_text, chunk_starts):
        for sent in split_sentences(txt):
            en = min(st + CHUNK_SEC, total_sec)
            srt_lines.append(f"{idx}\n{fmt_srt(st)} --> {fmt_srt(en)}\n{sent}\n")
            idx += 1
    open(base_path + ".srt", "w", encoding="utf-8").write("\n".join(srt_lines))
    full = "".join(chunks_text)
    paras, cur = [], ""
    for ch in full:
        cur += ch
        if len(cur) >= 160 and ch in "。！？":
            paras.append(cur); cur = ""
    if cur:
        paras.append(cur)
    open(base_path + ".txt", "w", encoding="utf-8").write("\n\n".join(paras) + "\n")

def process_one(enc, dec, tok, input_path, srt_dir):
    base = os.path.splitext(input_path)[0]
    srt_out = os.path.join(srt_dir, os.path.basename(base) + ".srt")
    txt_out = os.path.join(srt_dir, os.path.basename(base) + ".txt")
    if os.path.exists(srt_out) and os.path.getsize(srt_out) > 0:
        print(f"  [跳过] {os.path.basename(input_path)}")
        return
    wav = base + ".wav"
    if not os.path.exists(wav) or os.path.getsize(wav) < 1000:
        extract_audio(input_path, wav)
    audio = load_wav(wav)
    total_sec = len(audio) / SAMPLE_RATE
    print(f"  [{os.path.basename(input_path)}] {total_sec/60:.1f} 分钟", flush=True)
    t0 = time.time()
    texts, starts = [], []
    n = int(np.ceil(total_sec / CHUNK_SEC))
    for i in range(n):
        chunk = audio[i * SAMPLE_RATE * CHUNK_SEC:(i + 1) * SAMPLE_RATE * CHUNK_SEC]
        if len(chunk) < SAMPLE_RATE:
            break
        texts.append(transcribe_chunk(enc, dec, tok, chunk))
        starts.append(i * CHUNK_SEC)
    # 写 srt/txt 到 srt_dir（和音频同目录）
    srt_lines, idx = [], 1
    for txt, st in zip(texts, starts):
        for sent in split_sentences(txt):
            en = min(st + CHUNK_SEC, total_sec)
            srt_lines.append(f"{idx}\n{fmt_srt(st)} --> {fmt_srt(en)}\n{sent}\n")
            idx += 1
    open(srt_out, "w", encoding="utf-8").write("\n".join(srt_lines))
    full = "".join(texts)
    paras, cur = [], ""
    for ch in full:
        cur += ch
        if len(cur) >= 160 and ch in "。！？":
            paras.append(cur); cur = ""
    if cur:
        paras.append(cur)
    open(txt_out, "w", encoding="utf-8").write("\n\n".join(paras) + "\n")
    el = time.time() - t0
    print(f"  ✓ 完成 {el:.0f}s（{total_sec/el:.1f} 倍实时）", flush=True)

def main():
    parser = argparse.ArgumentParser(description="混合架构 Whisper 识别（GPU encoder + CPU decoder）")
    parser.add_argument("input", help="文件或目录")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--outdir", default=None, help="输出目录（默认与视频同目录）")
    args = parser.parse_args()

    print("[加载] GPU encoder + CPU decoder ...")
    enc = load_encoder(args.device)
    dec = load_decoder()
    tok = load_tokenizer()

    EXT = (".mp4", ".mkv", ".flv", ".avi", ".mov", ".wav", ".mp3", ".m4a")
    if os.path.isdir(args.input):
        files = []
        for root, _, fs in os.walk(args.input):
            for f in fs:
                if f.lower().endswith(EXT):
                    files.append(os.path.join(root, f))
        files.sort()
    else:
        files = [args.input]

    srt_dir = args.outdir or (os.path.dirname(os.path.abspath(files[0])) if files else ".")
    os.makedirs(srt_dir, exist_ok=True)
    print(f"共 {len(files)} 个文件，输出到 {srt_dir}")
    for fp in files:
        try:
            process_one(enc, dec, tok, fp, srt_dir)
        except Exception as e:
            print(f"  [错误] {os.path.basename(fp)}: {e}", flush=True)
    print("\n[完成]")

if __name__ == "__main__":
    main()
