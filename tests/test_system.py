# -*- coding: utf-8 -*-
"""
单元测试（pytest）
==================
覆盖：预处理、特征提取、自相关/基音检测、参数化模拟、智能 MLP 模拟、
评估指标、端点检测与端到端主流程（结题指标验证）。

运行方式（在项目根目录 voice/ 下）：
    python -m pytest tests -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vtsim import preprocess, features, simulate, neural_sim, evaluate  # noqa: E402
from vtsim.pipeline import process_audio, _check_feature_validity  # noqa: E402


# ==================== 测试数据：合成语音（正弦 + 噪声） ====================

def make_synthetic_signal(sr=16000, duration=2.0, f0=120.0):
    """
    合成分段语音（能量量级对齐 THCHS-30 真实语料，保证 MSE 阈值可比较）：
    浊音段（正弦，幅度 0.12）+ 清音段（噪声，σ=0.02）+ 静音段。
    """
    n = int(sr * duration)
    t = np.arange(n) / sr
    signal = np.zeros(n)
    seg1 = int(n * 0.40)   # 浊音段
    seg2 = int(n * 0.65)   # 浊音+清音结束
    voiced = (0.12 * np.sin(2 * np.pi * f0 * t)
              + 0.03 * np.sin(2 * np.pi * 2 * f0 * t))
    signal[0:seg1] = voiced[0:seg1]
    rng = np.random.default_rng(0)
    signal[seg1:seg2] = rng.normal(0, 0.02, seg2 - seg1)
    return signal.astype(np.float64)


def synthetic_features():
    """由合成语音提取时域特征（具有真实语音的时域结构，可复现）。"""
    signal = make_synthetic_signal()
    frames, _, _ = preprocess.framing(signal, 16000)
    frames = preprocess.apply_window(frames, "hamming")
    return features.extract_all_features(frames)


def write_wav(path, signal, sr=16000):
    import soundfile as sf
    sf.write(path, signal, sr)
    return path


@pytest.fixture(scope="module")
def synth_wav(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("wav")
    path = str(tmp / "synth.wav")
    write_wav(path, make_synthetic_signal())
    return path


# ==================== 1. 预处理 ====================

def test_pre_emphasis_shape():
    x = np.random.default_rng(1).normal(size=1000)
    y = preprocess.pre_emphasis(x, coeff=0.97)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_framing_counts():
    sr, dur = 16000, 1.0
    x = np.zeros(int(sr * dur))
    frames, fl, fs = preprocess.framing(x, sr, 25, 10)
    # 帧长 400 点，帧移 160 点；1 秒 = 16000 点
    assert fl == 400 and fs == 160
    expected = (16000 - 400) // 160 + 1
    assert len(frames) == expected
    assert frames.shape[1] == 400


def test_apply_window_energy_reduction():
    frames = np.ones((5, 400))
    assert abs(np.sum(preprocess.apply_window(frames, "rect")) - 2000) < 1e-6
    ham = np.sum(preprocess.apply_window(frames, "hamming"))
    assert ham < 2000          # 加窗后总能量降低


# ==================== 2. 特征提取 ====================

def test_short_time_energy_nonneg():
    x = np.random.default_rng(2).normal(size=400)
    e = features.short_time_energy(x)
    assert e >= 0
    assert abs(e - np.sum(x ** 2)) < 1e-8


def test_zero_crossing_rate_range():
    frames, _, _ = preprocess.framing(make_synthetic_signal(), 16000)
    zcrs = np.array([features.zero_crossing_rate(f) for f in frames])
    assert zcrs.min() >= 0.0 and zcrs.max() <= 1.0
    # 静音帧过零率接近 0
    silent = np.zeros(400)
    assert features.zero_crossing_rate(silent) == 0.0


def test_autocorrelation_normalized():
    rng = np.random.default_rng(3)
    x = rng.normal(size=400)
    acf = features.autocorrelation_normalized(x, max_lag=100)
    assert acf.shape == (100,)
    assert abs(acf[0] - 1.0) < 1e-6
    assert np.all(acf <= 1.0 + 1e-6) and np.all(acf >= -1.0 - 1e-6)


def test_detect_pitch_on_sine():
    sr = 16000
    t = np.arange(sr) / sr
    x = 0.8 * np.sin(2 * np.pi * 120 * t)
    frames, _, _ = preprocess.framing(x, sr, 25, 10)
    frames = preprocess.apply_window(frames, "hamming")
    f0, info = features.detect_pitch(frames, sr=sr)
    assert 100 < f0 < 145          # 应检出约 120 Hz


# ==================== 3. 参数化模拟 ====================

def test_param_simulation_metrics():
    real_e, real_z = synthetic_features()
    sim_e = simulate.generate_simulated_energies(real_e, seed=42)
    sim_z = simulate.generate_simulated_zcrs(real_z, seed=42)
    assert sim_e.shape == real_e.shape and sim_z.shape == real_z.shape
    mse_e, corr_e = evaluate.evaluate(real_e, sim_e)
    mse_z, corr_z = evaluate.evaluate(real_z, sim_z)
    assert mse_e <= 0.05 and corr_e >= 0.8    # 中期指标
    assert mse_z <= 0.05 and corr_z >= 0.8
    assert np.all(sim_e >= 0) and np.all((sim_z >= 0) & (sim_z <= 1))


# ==================== 4. 智能 MLP 模拟 ====================

def test_mlp_training_converges():
    rng = np.random.default_rng(5)
    real = np.abs(rng.normal(0.2, 0.3, 200)) + 1e-3
    sim, info = neural_sim.generate_mlp_sequence(
        real, seed=42, epochs=300)
    assert sim.shape == real.shape
    assert info["final_loss"] < info["initial_loss"]   # 损失确实下降
    assert info["final_loss"] < 0.5


def test_smart_simulation_metrics():
    real_e, real_z = synthetic_features()
    sim_e, _, info_e, _ = neural_sim.smart_simulate_features(
        real_e, real_z, seed=42)
    mse, corr = evaluate.evaluate(real_e, sim_e)
    assert mse <= 0.03 and corr >= 0.85            # 结题指标
    assert info_e["final_loss"] < info_e["initial_loss"]


# ==================== 5. 评估 ====================

def test_evaluate_perfect_match():
    x = np.linspace(0, 1, 100)
    mse, corr = evaluate.evaluate(x, x)
    assert mse == pytest.approx(0.0, abs=1e-12)
    assert corr == pytest.approx(1.0, abs=1e-9)


def test_feature_validity_selfcheck():
    e = np.array([0.1, 0.2, -0.05, 1.5, 0.3])      # 一个负能量样本
    z = np.array([0.1, 0.2, 0.3, 1.2, 0.5])        # 一个越界样本
    result, ratio = _check_feature_validity(e, z)
    assert result["valid_ratio"] < 1.0
    assert ratio <= 0.96


# ==================== 6. 端点检测 ====================

def test_voice_activity_detection():
    # 静音-语音-静音-语音结构
    energy = np.zeros(100)
    energy[20:50] = 1.0
    energy[60:80] = 0.8
    mask, bounds = preprocess.voice_activity_detection(energy)
    assert len(bounds) == 2
    assert mask[20] and mask[49] and mask[60] and mask[79]
    assert not mask[0] and not mask[10] and not mask[55] and not mask[90]


# ==================== 7. 端到端主流程（结题指标验证） ====================

def test_pipeline_end_to_end(synth_wav, tmp_path):
    result = process_audio(synth_wav, {
        "export": False, "show_plot": False, "vad": True, "seed": 42,
    }, verbose=False)
    assert result["ok"]
    assert result["n_frames"] > 100
    m = result["metrics"]
    assert m["final_pass_energy"] and m["final_pass_zcr"]
    assert m["mid_pass_energy"] and m["mid_pass_zcr"]
    assert result["self_check"]["valid_ratio"] >= 0.95
    # 导出的 PNG/CSV/TXT 存在
    out = str(tmp_path / "out")
    result2 = process_audio(synth_wav, {
        "export": True, "show_plot": False, "outdir": out,
    }, verbose=False)
    for kind in ("png", "csv", "txt"):
        assert os.path.exists(result2["files"][kind]), kind
        assert os.path.getsize(result2["files"][kind]) > 0, kind


# ==================== 8. 可复现性 ====================

def test_reproducibility_with_seed():
    rng = np.random.default_rng(7)
    real = np.abs(rng.normal(0.2, 0.3, 200)) + 1e-3
    s1, _ = neural_sim.generate_mlp_sequence(real, seed=123, epochs=200)
    s2, _ = neural_sim.generate_mlp_sequence(real, seed=123, epochs=200)
    assert np.allclose(s1, s2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))