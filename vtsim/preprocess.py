# -*- coding: utf-8 -*-
"""
预处理模块
==========
实现：音频加载、预加重、分帧、加窗、谱减法降噪、双门限端点检测。
参数全部可自定义，满足“预处理参数可自定义”的结题功能目标。

说明：
    分帧采用“短时平稳性”假设：语音信号在 10~30 ms 内近似平稳，
    帧长 20~30 ms、帧移 10~15 ms 为语音信号处理常用参数。
"""

import numpy as np


# ==================== 音频加载 ====================

def load_audio(path, sr=16000):
    """
    加载音频文件（支持 wav/mp3/flac 等格式，自动转为单声道、重采样到 sr）。
    返回 (signal, sr)，signal 为归一化浮点数组（取值范围约 -1 ~ 1）。
    """
    import librosa
    signal, sr_ = librosa.load(path, sr=sr, mono=True)
    return signal, int(sr_)


# ==================== 预加重 ====================

def pre_emphasis(signal, coeff=0.97):
    """
    预加重：一阶高通滤波 y(n) = x(n) - alpha*x(n-1)。
    用于补偿语音产生过程中高频能量的自然衰减，使频谱更平坦，
    提升后续特征提取（尤其过零率）对高频细节的敏感度。
    """
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) < 2:
        return signal.copy()
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


# ==================== 分帧 ====================

def framing(signal, sr, frame_len_ms=25, frame_shift_ms=10):
    """
    分帧：将连续信号切分为重叠的短时帧。
    帧长 frame_len_ms（默认 25 ms = sr*0.025 个采样点），
    帧移 frame_shift_ms（默认 10 ms）。
    返回 (frames, frame_len, frame_shift)，frames 形状为 (num_frames, frame_len)。
    """
    signal = np.asarray(signal, dtype=np.float64)
    frame_len = int(round(sr * frame_len_ms / 1000.0))
    frame_shift = int(round(sr * frame_shift_ms / 1000.0))
    frame_len = max(frame_len, 8)
    frame_shift = max(frame_shift, 1)

    if len(signal) < frame_len:
        raise ValueError(
            f"信号过短（{len(signal)} 个采样点），小于一帧长度 {frame_len}，"
            f"请减小帧长或使用更长的音频。")

    num_frames = (len(signal) - frame_len) // frame_shift + 1
    frames = np.stack([signal[i * frame_shift: i * frame_shift + frame_len]
                       for i in range(num_frames)])
    return frames, frame_len, frame_shift


# ==================== 加窗 ====================

def apply_window(frames, window_type="hamming"):
    """
    加窗：对每一帧乘窗函数，平滑帧两端边缘，抑制频谱泄漏。
    window_type: 'hamming'（汉明窗，默认）、'hann'（汉宁窗）、'rect'（矩形窗=不加窗）。
    """
    frame_len = frames.shape[1]
    if window_type == "hamming":
        window = np.hamming(frame_len)
    elif window_type == "hann":
        window = np.hanning(frame_len)
    else:  # rect
        window = np.ones(frame_len)
    return frames * window


# ==================== 谱减法降噪 ====================

def spectral_subtraction(signal, sr, n_fft=512, hop_length=None,
                         noise_duration=0.25, over_subtraction=1.5,
                         noise_floor=0.02):
    """
    轻量谱减法降噪（对应中期报告“结合谱减法估计抵消噪声”的对策）。
    1) 取信号前 noise_duration 秒（或整段中能量最低的帧）估计噪声幅度谱；
    2) STFT -> 幅度谱减去估计噪声谱（过减因子 over_subtraction）；
    3) 幅度下限限制（noise_floor），保留原相位，ISTFT 还原信号。

    适用于背景噪声平稳的场景；纯静音信号不会被误伤（幅度已被下限保护）。
    """
    import librosa
    signal = np.asarray(signal, dtype=np.float64)
    hop_length = hop_length or n_fft // 4

    # 若信号太短则直接返回
    if len(signal) < n_fft * 2:
        return signal.copy()

    # 用前 noise_duration 秒估计噪声幅度谱；若过短则用全段最低能量帧
    n_noise = max(int(sr * noise_duration), n_fft)
    noise_seg = signal[:min(n_noise, len(signal))]
    noise_stft = librosa.stft(noise_seg, n_fft=n_fft, hop_length=hop_length)
    noise_mag = np.abs(noise_stft).mean(axis=1, keepdims=True)

    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft)
    phase = np.angle(stft)

    # 谱减法（含过减与幅度下限）
    clean_mag = np.maximum(mag - over_subtraction * noise_mag,
                           noise_floor * mag)

    clean_stft = clean_mag * np.exp(1j * phase)
    clean = librosa.istft(clean_stft, hop_length=hop_length,
                          length=len(signal))
    return clean


# ==================== 双门限端点检测 ====================

def voice_activity_detection(energies, zcrs=None,
                             sil_ratio=0.10, speech_ratio=0.40,
                             min_silence_frames=5, min_speech_frames=3):
    """
    经典双门限端点检测（VAD）：
        * 高门限 T_high：高于它判定进入语音段；
        * 低门限 T_low ：低于它且持续 min_silence_frames 帧判定语音段结束；
        低于 min_speech_frames 帧的碎片语音段被合并/丢弃，避免毛刺误判。

    参数：
        energies       : 每帧短时能量
        zcrs           : 每帧短时过零率（可选，用于清音扩展）
        sil_ratio      : 低门限比例（相对能量动态范围）
        speech_ratio   : 高门限比例
    返回：
        (speech_mask, boundaries)
        speech_mask : bool 数组，True 表示语音帧
        boundaries  : [(start, end), ...] 语音段（帧序号，左闭右开）
    """
    energies = np.asarray(energies, dtype=np.float64)
    n = len(energies)
    if n == 0:
        return np.zeros(0, dtype=bool), []

    e_min = float(energies.min())
    e_max = float(energies.max())
    if e_max <= e_min:
        return np.zeros(n, dtype=bool), []

    t_low = e_min + (e_max - e_min) * sil_ratio
    t_high = e_min + (e_max - e_min) * speech_ratio

    mask = np.zeros(n, dtype=bool)
    in_speech = False
    sil_count = 0

    for i in range(n):
        e = energies[i]
        above_high = e >= t_high
        above_low = e >= t_low

        if not in_speech:
            if above_high:
                mask[i] = True
                in_speech = True
                sil_count = 0
            # 清音段扩展：过零率高的帧也可并入语音（可选）
            elif zcrs is not None and e >= t_low * 0.5 and zcrs[i] > 0.4 * (zcrs.max() if zcrs.max() > 0 else 1):
                mask[i] = True
                in_speech = True
                sil_count = 0
        else:
            in_speech_now = above_low
            if in_speech_now:
                mask[i] = True
                sil_count = 0
            else:
                sil_count += 1
                if sil_count > min_silence_frames:
                    in_speech = False
                    sil_count = 0
                else:
                    mask[i] = True  # 短静音并入语音段

    # 丢弃过短的语音碎片
    boundaries = []
    start = None
    for i in range(n + 1):
        is_speech = i < n and mask[i]
        if is_speech and start is None:
            start = i
        elif not is_speech and start is not None:
            if i - start >= min_speech_frames:
                boundaries.append((start, i))
            else:
                mask[start:i] = False
            start = None
    return mask, boundaries