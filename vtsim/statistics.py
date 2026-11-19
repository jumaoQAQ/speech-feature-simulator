# -*- coding: utf-8 -*-
"""
统计分析模块
============
实现：特征统计量计算、概率分布拟合（正态/伽马/指数）与拟合优度评价。
中期实验结论：短时能量基本服从正态分布，短时过零率在短语音中服从伽马分布、
长语音中趋近正态分布。该模块负责定量复现并输出这些结论。

scipy 为可选依赖：缺失时自动退化为仅正态分布拟合（numpy 实现）。
"""

import numpy as np

try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _scipy_stats = None
    _HAS_SCIPY = False


# ==================== 分布拟合 ====================

def _norm_pdf(x, mu, sigma):
    """numpy 实现的正态分布概率密度（无 scipy 时的备选）。"""
    sigma = max(float(sigma), 1e-8)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _kde_mse_numpy(data, fitted_pdf, x_axis):
    """用核密度估计(KDE)与拟合分布 pdf 的均方误差评价拟合优度（无 scipy 备选）。"""
    data = np.asarray(data, dtype=np.float64)
    h = 1.06 * np.std(data) * len(data) ** (-1.0 / 5.0)  # Scott 规则带宽
    h = max(h, 1e-6)
    kde = np.zeros_like(x_axis)
    for d in data:
        kde += np.exp(-0.5 * ((x_axis - d) / h) ** 2) / (h * np.sqrt(2 * np.pi))
    kde /= len(data)
    return float(np.mean((fitted_pdf - kde) ** 2))


def fit_distribution(data):
    """
    拟合数据的最佳概率分布（正态/伽马/指数），以 KDE 拟合误差 MSE 最小为准则。
    返回 (best_name, best_params, mse)。参数格式（便于打印）：
        正态: (mean, std)；伽马: (shape, loc, scale)；指数: (loc, scale)
    """
    data = np.asarray(data, dtype=np.float64)
    data = data[data > 0]                      # 分布拟合只取正样本
    if len(data) < 10:
        return "正态分布", (float(np.mean(data)), float(np.std(data))), float("inf")

    if _HAS_SCIPY:
        candidates = {
            "正态分布": _scipy_stats.norm,
            "伽马分布": _scipy_stats.gamma,
            "指数分布": _scipy_stats.expon,
        }
    else:  # pragma: no cover - scipy 缺失时的退化路径
        candidates = None

    best_name, best_params, best_mse = "正态分布", (
        float(np.mean(data)), float(np.std(data))), float("inf")

    x_axis = np.linspace(float(data.min()), float(data.max()), 100)

    if candidates is None:  # pragma: no cover
        mu, sigma = np.mean(data), np.std(data)
        mse = _kde_mse_numpy(data, _norm_pdf(x_axis, mu, sigma), x_axis)
        return "正态分布", (float(mu), float(sigma)), float(mse)

    for name, dist in candidates.items():
        try:
            params = dist.fit(data)
            pdf_fitted = dist.pdf(x_axis, *params)
            kde_raw = _scipy_stats.gaussian_kde(data)(x_axis)
            mse = float(np.mean((pdf_fitted - kde_raw) ** 2))
            if mse < best_mse:
                best_mse, best_name, best_params = mse, name, params
        except Exception:
            continue

    # 统一参数为元组（浮点数），便于 JSON/打印
    best_params = tuple(float(p) for p in best_params)
    return best_name, best_params, best_mse


def _fmt_params(name, params):
    """把分布参数格式化为可读字符串。"""
    p = tuple(float(x) for x in params)
    if name == "正态分布" and len(p) >= 2:
        return f"μ={p[0]:.4f}, σ={p[1]:.4f}"
    if name == "伽马分布" and len(p) >= 3:
        return f"α(shape)={p[0]:.4f}, loc={p[1]:.4f}, β(scale)={p[2]:.4f}"
    if name == "指数分布" and len(p) >= 2:
        return f"loc={p[0]:.4f}, β(scale)={p[1]:.4f}"
    return str(p)


# ==================== 特征统计分析 ====================

def analyze_features(energies, zcrs):
    """
    分析短时能量与短时过零率的统计特性，返回统计字典。
    键：energy_mean/energy_std/energy_dist/energy_dist_params/zcr_* ...
    """
    energies = np.asarray(energies, dtype=np.float64)
    zcrs = np.asarray(zcrs, dtype=np.float64)

    energy_dist, energy_params, energy_mse = fit_distribution(energies)
    zcr_dist, zcr_params, zcr_mse = fit_distribution(zcrs)

    return {
        "energy_count": int(len(energies)),
        "energy_mean": float(np.mean(energies)),
        "energy_std": float(np.std(energies)),
        "energy_min": float(np.min(energies)),
        "energy_max": float(np.max(energies)),
        "energy_dist": energy_dist,
        "energy_dist_params": energy_params,
        "energy_fit_mse": float(energy_mse),
        "energy_dist_str": f"{energy_dist} ({_fmt_params(energy_dist, energy_params)})",
        "zcr_count": int(len(zcrs)),
        "zcr_mean": float(np.mean(zcrs)),
        "zcr_std": float(np.std(zcrs)),
        "zcr_min": float(np.min(zcrs)),
        "zcr_max": float(np.max(zcrs)),
        "zcr_dist": zcr_dist,
        "zcr_dist_params": zcr_params,
        "zcr_fit_mse": float(zcr_mse),
        "zcr_dist_str": f"{zcr_dist} ({_fmt_params(zcr_dist, zcr_params)})",
    }


def format_stats_report(stats_info):
    """将统计字典格式化为多行可打印文本。"""
    lines = []
    lines.append("【短时能量】")
    lines.append(f"  样本数: {stats_info['energy_count']}")
    lines.append(f"  均值: {stats_info['energy_mean']:.6f}   标准差: {stats_info['energy_std']:.6f}")
    lines.append(f"  最小值: {stats_info['energy_min']:.6f}   最大值: {stats_info['energy_max']:.6f}")
    lines.append(f"  最佳拟合分布: {stats_info['energy_dist_str']}")
    lines.append("")
    lines.append("【短时过零率】")
    lines.append(f"  样本数: {stats_info['zcr_count']}")
    lines.append(f"  均值: {stats_info['zcr_mean']:.6f}   标准差: {stats_info['zcr_std']:.6f}")
    lines.append(f"  最小值: {stats_info['zcr_min']:.6f}   最大值: {stats_info['zcr_max']:.6f}")
    lines.append(f"  最佳拟合分布: {stats_info['zcr_dist_str']}")
    return "\n".join(lines)