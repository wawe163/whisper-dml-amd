# -*- coding: utf-8 -*-
"""
Whisper 的 log-mel 频谱计算（独立实现，无重依赖）
严格复刻 OpenAI Whisper 的 FeatureExtractor 逻辑：
  - n_fft=400, hop_length=160, 16kHz, 80 mel bins
  - Slaney 风格 mel 滤波器组
  - log10 后 clamp 到 max-8，再 (x+4)/4 归一化
"""
import numpy as np

N_FFT = 400
HOP = 160
N_MELS = 80
SAMPLE_RATE = 16000
NB_MAX_FRAMES = 3000  # 30秒 = 480000采样 / 160 = 3000帧

def mel_filters():
    """Slaney 风格 mel 滤波器组，返回 [80, 201]"""
    n_fft = N_FFT
    sr = SAMPLE_RATE
    n_mels = N_MELS

    # 每个 FFT bin 的中心频率
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

    # mel 频段的中心频率（Slaney 风格，linear + log 混合）
    min_mel = 0.0
    max_mel = 45.245640471924965
    mels = np.linspace(min_mel, max_mel, n_mels + 2)

    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

    # 三角滤波器权重
    fdiff = np.diff(freqs)
    ramps = freqs.reshape(-1, 1) - fftfreqs.reshape(1, -1)
    lower = -ramps[:-2] / np.expand_dims(fdiff[:-1], axis=1)
    upper = ramps[2:] / np.expand_dims(fdiff[1:], axis=1)
    weights = np.maximum(np.zeros_like(lower), np.minimum(lower, upper))

    # Slaney 风格缩放（近似恒定能量）
    enorm = 2.0 / (freqs[2:n_mels + 2] - freqs[:n_mels])
    weights *= np.expand_dims(enorm, axis=1)

    return weights.astype(np.float32)

_MEL_FILTERS = None

def log_mel_spectrogram(audio):
    """
    输入: float32 数组（16kHz，归一化到 [-1,1]）
    输出: [80, T] log-mel 频谱（T 接近 3000）
    """
    global _MEL_FILTERS
    if _MEL_FILTERS is None:
        _MEL_FILTERS = mel_filters()

    n_fft = N_FFT
    hop = HOP

    # 反射 padding（center=True, mode=reflect）
    audio = np.pad(audio, (0, n_fft // 2), mode="reflect")

    n_frames = 1 + (len(audio) - n_fft) // hop
    # 用 stride 取帧
    frames = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::hop][:n_frames]

    # Hann 窗
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    frames = frames * window

    # STFT 幅度平方
    spec = np.fft.rfft(frames, axis=1)
    mag = np.abs(spec) ** 2

    # mel 映射
    mel = mag @ _MEL_FILTERS.T
    mel = np.log10(np.maximum(mel, 1e-10))
    mel = np.maximum(mel, mel.max() - 8.0)
    mel = (mel + 4.0) / 4.0

    return mel.T.astype(np.float32)  # [80, T]


def pad_or_trim(mel):
    """填充/截断到 3000 帧"""
    T = mel.shape[1]
    if T > NB_MAX_FRAMES:
        return mel[:, :NB_MAX_FRAMES]
    if T < NB_MAX_FRAMES:
        pad = np.zeros((mel.shape[0], NB_MAX_FRAMES - T), dtype=np.float32)
        return np.concatenate([mel, pad], axis=1)
    return mel
