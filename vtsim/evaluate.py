# -*- coding: utf-8 -*-
"""
评估模块
========
实现：MSE / 皮尔逊相关系数 / RMSE / 分布重叠度等指标，以及
中期指标（MSE≤0.05、r≥0.8）与结题指标（MSE≤0.03、r≥0.85）判定。
"""

import numpy as np


# 结题验收指标（立项报告“（四）结题预期目标”）
FINAL_MSE_LIMIT = 0.03
FINAL_CORR_LIMIT = 0.85
# 中期指标（用于对比展示）
MID_MSE_LIMIT = 0.05
MID_CORR_LIMIT = 0.80


def evaluate(real, sim):
    """
    计算 (MSE, 皮尔逊相关系数)。
    截断模拟序列到与真实序列等长；长度不足 2 时相关系数置 0。
    """
    real = np.asarray(real, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)[:len(real)]
    mse = float(np.mean((real - sim) ** 2))
    if len(real) < 2 or np.std(real) < 1e-12 or np.std(sim) < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(real, sim)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
    return mse, corr


def evaluate_features(real_energy, param_energy, smart_energy,
                      real_zcr, param_zcr, smart_zcr):
    """
    汇总评估：分别计算参数化模拟与智能模拟的 (MSE, r)。
    返回结构化的评估字典，包含判定结果（结题指标 / 中期指标）。
    """
    pe_mse, pe_corr = evaluate(real_energy, param_energy)
    se_mse, se_corr = evaluate(real_energy, smart_energy)
    pz_mse, pz_corr = evaluate(real_zcr, param_zcr)
    sz_mse, sz_corr = evaluate(real_zcr, smart_zcr)

    def _judge(mse, corr, mse_lim, corr_lim):
        return bool(mse <= mse_lim and corr >= corr_lim)

    return {
        "param_energy_mse": pe_mse,
        "param_energy_corr": pe_corr,
        "smart_energy_mse": se_mse,
        "smart_energy_corr": se_corr,
        "param_zcr_mse": pz_mse,
        "param_zcr_corr": pz_corr,
        "smart_zcr_mse": sz_mse,
        "smart_zcr_corr": sz_corr,
        # 结题指标判定（智能模拟，核心验收项）
        "final_pass_energy": _judge(se_mse, se_corr,
                                    FINAL_MSE_LIMIT, FINAL_CORR_LIMIT),
        "final_pass_zcr": _judge(sz_mse, sz_corr,
                                 FINAL_MSE_LIMIT, FINAL_CORR_LIMIT),
        "final_pass_all": _judge(se_mse, se_corr,
                                 FINAL_MSE_LIMIT, FINAL_CORR_LIMIT)
                          and _judge(sz_mse, sz_corr,
                                     FINAL_MSE_LIMIT, FINAL_CORR_LIMIT),
        # 中期指标判定（参数化模拟，对比展示）
        "mid_pass_energy": _judge(pe_mse, pe_corr,
                                  MID_MSE_LIMIT, MID_CORR_LIMIT),
        "mid_pass_zcr": _judge(pz_mse, pz_corr,
                               MID_MSE_LIMIT, MID_CORR_LIMIT),
        "mid_pass_all": _judge(pe_mse, pe_corr,
                               MID_MSE_LIMIT, MID_CORR_LIMIT)
                        and _judge(pz_mse, pz_corr,
                                   MID_MSE_LIMIT, MID_CORR_LIMIT),
    }


def format_evaluation_report(e):
    """将评估字典格式化为多行可打印文本（含达标/未达标标记）。"""
    lines = []
    lines.append("=" * 68)
    lines.append("评估报告（模拟效果指标）")
    lines.append("=" * 68)
    lines.append("结题验收指标：智能模拟 MSE≤0.03、相关系数≥0.85")
    lines.append("中期指标（参考）：参数化模拟 MSE≤0.05、相关系数≥0.80")
    lines.append("")

    rows = [
        ("短时能量（参数化模拟）", "param", "mid", e["param_energy_mse"], e["param_energy_corr"],
         e["mid_pass_energy"]),
        ("短时能量（智能MLP模拟）", "smart", "final", e["smart_energy_mse"], e["smart_energy_corr"],
         e["final_pass_energy"]),
        ("短时过零率（参数化模拟）", "param", "mid", e["param_zcr_mse"], e["param_zcr_corr"],
         e["mid_pass_zcr"]),
        ("短时过零率（智能MLP模拟）", "smart", "final", e["smart_zcr_mse"], e["smart_zcr_corr"],
         e["final_pass_zcr"]),
    ]
    lines.append(f"{'项目':<22}{'MSE':>12}{'相关系数':>12}{'判定':>10}")
    lines.append("-" * 68)
    for name, _, _, mse, corr, ok in rows:
        mark = "达标" if ok else "未达标"
        lines.append(f"{name:<22}{mse:>12.6f}{corr:>12.6f}{mark:>10}")

    lines.append("")
    lines.append("【结题指标总体判定（智能模拟）】")
    lines.append(f"  短时能量: {'达标' if e['final_pass_energy'] else '未达标'}"
                 f" (MSE={e['smart_energy_mse']:.6f}, r={e['smart_energy_corr']:.6f})")
    lines.append(f"  短时过零率: {'达标' if e['final_pass_zcr'] else '未达标'}"
                 f" (MSE={e['smart_zcr_mse']:.6f}, r={e['smart_zcr_corr']:.6f})")
    lines.append(f"  总体结论: {'全部达标' if e['final_pass_all'] else '尚未全部达标'}")
    return "\n".join(lines)