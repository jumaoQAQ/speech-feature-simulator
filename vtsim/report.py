# -*- coding: utf-8 -*-
"""
可视化与导出模块
================
实现：单窗口结果图（PNG）、特征数据 CSV、文本报告 TXT。
单窗口集成 6 个子图，方便汇报场景一键截图（延续中期版本的设计）。
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无 GUI 环境也保证可保存图片（exe 打包必需）
import matplotlib.pyplot as plt

from .features import autocorrelation_normalized

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei",
                                   "PingFang SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ==================== 单窗口可视化 ====================

def plot_all_in_one(real_energy, sim_energy_param, sim_energy_smart,
                    real_zcr, sim_zcr_param, sim_zcr_smart,
                    frames, pitch_info=None, save_path=None,
                    title="语音短时时域特征模拟系统 - 结题结果展示"):
    """
    2 行 3 列单窗口结果图：
        (a) 短时能量时序对比（真实/参数化/智能）
        (b) 能量相关性散点（真实 vs 智能）
        (c) 浊音帧短时自相关函数（基音检测）
        (d) 短时过零率时序对比（真实/参数化/智能）
        (e) 过零率相关性散点（真实 vs 智能）
        (f) 能量分布对比直方图（真实/智能）
    返回 matplotlib Figure 对象；save_path 非空时同时保存 PNG。
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    n = len(real_energy)
    step = max(1, n // 300)

    # (a) 能量时序对比
    ax = axes[0, 0]
    ax.plot(real_energy[::step], label="真实能量", linewidth=1.5)
    ax.plot(sim_energy_param[::step], label="参数化模拟", linewidth=1.2,
            alpha=0.65)
    ax.plot(sim_energy_smart[::step], label="智能模拟(MLP)", linewidth=1.2,
            alpha=0.65)
    ax.set_title("(a) 短时能量时序对比", fontsize=12)
    ax.set_xlabel("帧序号")
    ax.set_ylabel("能量值")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) 能量相关性散点
    ax = axes[0, 1]
    ax.scatter(real_energy, sim_energy_smart, alpha=0.3, s=2)
    lo = min(real_energy.min(), sim_energy_smart.min())
    hi = max(real_energy.max(), sim_energy_smart.max())
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    ax.plot([lo, hi], [lo, hi], "r--", alpha=0.5, linewidth=1.5)
    corr = np.corrcoef(real_energy, sim_energy_smart)[0, 1]
    ax.set_title(f"(b) 能量相关性（智能模拟, r={corr:.4f}）", fontsize=12)
    ax.set_xlabel("真实能量")
    ax.set_ylabel("模拟能量")
    ax.grid(True, alpha=0.3)

    # (c) 自相关函数（浊音帧）
    ax = axes[0, 2]
    if len(frames) > 0:
        energies = np.array([np.sum(f ** 2) for f in frames])
        voiced_idx = int(np.argmax(energies))
        acf = autocorrelation_normalized(frames[voiced_idx], max_lag=300)
        ax.plot(acf)
        if pitch_info and pitch_info.get("lag"):
            ax.axvline(pitch_info["lag"], color="r", linestyle="--",
                       linewidth=1.2)
            ax.annotate(f"基音周期={pitch_info['lag']}点\n"
                        f"(f0={pitch_info.get('f0_hz', 0):.1f}Hz)",
                        xy=(pitch_info["lag"], acf[pitch_info["lag"]]),
                        xytext=(pitch_info["lag"] + 40, 0.4),
                        arrowprops=dict(arrowstyle="->", color="red"),
                        fontsize=10, color="red")
    ax.set_title("(c) 浊音帧自相关函数（基音检测）", fontsize=12)
    ax.set_xlabel("延迟 (lag)")
    ax.set_ylabel("自相关值")
    ax.set_ylim(-0.5, 1.05)
    ax.grid(True, alpha=0.3)

    # (d) 过零率时序对比
    ax = axes[1, 0]
    ax.plot(real_zcr[::step], label="真实过零率", linewidth=1.5)
    ax.plot(sim_zcr_param[::step], label="参数化模拟", linewidth=1.2,
            alpha=0.65)
    ax.plot(sim_zcr_smart[::step], label="智能模拟(MLP)", linewidth=1.2,
            alpha=0.65)
    ax.set_title("(d) 短时过零率时序对比", fontsize=12)
    ax.set_xlabel("帧序号")
    ax.set_ylabel("过零率")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (e) 过零率相关性散点
    ax = axes[1, 1]
    ax.scatter(real_zcr, sim_zcr_smart, alpha=0.3, s=2)
    ax.plot([0, 1], [0, 1], "r--", alpha=0.5, linewidth=1.5)
    zcorr = np.corrcoef(real_zcr, sim_zcr_smart)[0, 1]
    ax.set_title(f"(e) 过零率相关性（智能模拟, r={zcorr:.4f}）", fontsize=12)
    ax.set_xlabel("真实过零率")
    ax.set_ylabel("模拟过零率")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # (f) 能量分布对比
    ax = axes[1, 2]
    ax.hist(real_energy, bins=50, alpha=0.5, label="真实能量", density=True)
    ax.hist(sim_energy_smart, bins=50, alpha=0.5, label="智能模拟能量",
            density=True)
    ax.set_title("(f) 能量分布对比（真实 vs 智能模拟）", fontsize=12)
    ax.set_xlabel("能量值")
    ax.set_ylabel("概率密度")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ==================== CSV 导出 ====================

def save_features_csv(path, real_energy, param_energy, smart_energy,
                      real_zcr, param_zcr, smart_zcr):
    """
    导出特征数据 CSV：
    帧序号 | 真实能量 | 参数化能量 | 智能能量 | 真实过零率 | 参数化过零率 | 智能过零率
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = len(real_energy)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["帧序号", "真实能量", "参数化模拟能量", "智能模拟能量",
                         "真实过零率", "参数化模拟过零率", "智能模拟过零率"])
        for i in range(n):
            writer.writerow([i + 1,
                             f"{real_energy[i]:.8f}", f"{param_energy[i]:.8f}",
                             f"{smart_energy[i]:.8f}", f"{real_zcr[i]:.8f}",
                             f"{param_zcr[i]:.8f}", f"{smart_zcr[i]:.8f}"])
    return path


# ==================== TXT 报告导出 ====================

def save_text_report(path, sections):
    """
    将多个文本段依次写入 TXT 报告。
    sections: [(标题, 正文多行文本), ...]
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8-sig") as f:
        for title, body in sections:
            f.write("=" * 68 + "\n")
            f.write(title + "\n")
            f.write("=" * 68 + "\n")
            f.write(body + "\n\n")
    return path