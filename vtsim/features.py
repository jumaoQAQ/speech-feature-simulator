# -*- coding: utf-8 -*-
"""
特征提取模块
============
实现：短时能量、短时过零率、短时自相关函数与基音检测（双阈值机制）。
三大时域特征是语音信号处理的基础：
    短时能量  —— 反映帧内响度，用于有声/静音、浊/清音判别；
    短时过零率 —— 反映波形穿过零电平的次数，清音/噪声显著高于浊音；
    短时自相关 —— 反映信号的周期性，浊音在基音周期处出现明显峰值。
"""

import numpy as np


# ==================== 短时能量 ====================

def short_time_energy(frame):
    """短时能量：E = sum(x(n)^2)。"""
    frame = np.asarray(frame, dtype=np.float64)
    return float(np.sum(frame ** 2))


# ==================== 短时过零率 ====================

def zero_crossing_rate(frame):
    """
    短时过零率：ZCR = 1/(2N) * sum(|sign(x(n)) - sign(x(n-1))|)。
    结果范围 [0, 1]。
    """
    frame = np.asarray(frame, dtype=np.float64)
    if len(frame) < 2:
        return 0.0
    crosses = np.abs(np.sign(frame[1:]) - np.sign(frame[:-1]))
    return float(np.sum(crosses) / (2.0 * (len(frame) - 1)))


# ==================== 批量提取 ====================

def extract_all_features(frames):
    """提取所有帧的短时能量与短时过零率，返回两个一维数组。"""
    frames = np.asarray(frames, dtype=np.float64)
    energies = np.array([short_time_energy(f) for f in frames])
    zcrs = np.array([zero_crossing_rate(f) for f in frames])
    return energies, zcrs


# ==================== 短时自相关 ====================

def autocorrelation_normalized(x, max_lag=None):
    """
    归一化短时自相关函数（去均值后计算，结果范围约 [-1, 1]）：
        R(k) = sum_n (x(n)-m)(x(n+k)-m) / sum_n (x(n)-m)^2
    其中 R(0)=1，峰值位置对应基音周期。
    """
    x = np.asarray(x, dtype=np.float64)
    if max_lag is None:
        max_lag = len(x)
    max_lag = int(min(max_lag, len(x)))
    if len(x) < 2 or max_lag < 1:
        return np.array([1.0])

    x = x - np.mean(x)
    r_full = np.correlate(x, x, mode="full")
    start = len(x) - 1                      # 延迟 0 的位置
    r = r_full[start: start + max_lag].astype(np.float64)
    r0 = r[0]
    if r0 > 1e-8:
        r = r / r0
    else:
        r = np.zeros_like(r)
        r[0] = 1.0
    return r


# ==================== 基音检测（双阈值机制） ====================

def detect_pitch(frames, sr=16000, f_min=60, f_max=500,
                 voicing_threshold=0.30, max_candidates=20):
    """
    基于自相关的基音检测（对应中期报告“引入双阈值基音检测机制”的对策）：
        1) 取能量最高的 max_candidates 帧作为浊音候选帧（能量阈值筛选）；
        2) 在基音频率范围 [f_min, f_max] 对应的延迟区间内搜索自相关峰值
           （峰值必须超过 voicing_threshold 才判定为浊音——第二阈值）；
        3) 对多个候选帧的基音周期取中位数，得到稳健的基音估计。

    返回 (f0_hz, info)：
        f0_hz : 基音频率（Hz）；无有效浊音帧时返回 0
        info  : 包含 lag、peak、voiced_ratio 等信息的字典（供报告展示）
    """
    frames = np.asarray(frames, dtype=np.float64)
    n_frames, frame_len = frames.shape
    lo = max(int(sr / f_max), 2)                 # 最小延迟
    hi = min(int(sr / f_min), frame_len - 1)     # 最大延迟
    if hi <= lo or n_frames == 0:
        return 0.0, {"lag": 0, "peak": 0.0, "voiced_ratio": 0.0}

    energies = np.array([short_time_energy(f) for f in frames])
    voiced_ratio = float(np.mean(energies > np.quantile(energies, 0.5)))

    # 候选浊音帧：能量最高的前 max_candidates 帧
    cand_indices = np.argsort(energies)[-min(max_candidates, n_frames):]
    lags, peaks = [], []
    for idx in cand_indices:
        acf = autocorrelation_normalized(frames[idx], max_lag=hi + 1)
        seg = acf[lo:hi + 1]
        if seg.size == 0:
            continue
        p = lo + int(np.argmax(seg))
        peak = float(acf[p])
        if peak >= voicing_threshold:            # 双阈值之一：清/浊判别
            lags.append(p)
            peaks.append(peak)

    if not lags:
        return 0.0, {"lag": 0, "peak": 0.0,
                     "voiced_ratio": voiced_ratio, "method": "自相关(双阈值)"}

    lag = int(np.median(lags))
    f0 = sr / lag
    info = {
        "lag": lag,
        "f0_hz": float(f0),
        "peak": float(np.mean(peaks)),
        "voiced_ratio": voiced_ratio,
        "method": "自相关(双阈值)",
    }
    return float(f0), info