# -*- coding: utf-8 -*-
"""
语音短时时域特征模拟系统 - 交互式界面（Streamlit）
====================================================
对应立项报告“基于 Streamlit 搭建图形界面，实现图表、数据导出”的结题任务。

启动方式：
    streamlit run streamlit_app.py

界面功能：
    * 侧边栏自定义预处理参数（采样率/帧长/帧移/窗函数/预加重系数/降噪/种子等）
    * 上传本地音频，或直接选择 test1/test2/test3 示例音频
    * 一键运行完整流程，展示统计报告、评估指标、单窗口结果图
    * 下载导出 CSV（特征数据）、PNG（结果图）、TXT（完整报告）
"""

import os
import sys
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from vtsim.pipeline import process_audio, DEFAULT_PARAMS  # noqa: E402
from vtsim import APP_NAME, __version__  # noqa: E402


EXAMPLES = {
    "test1.wav（标准基准样本）": "test1.wav",
    "test2.wav（自制实测样本）": "test2.wav",
    "test3.wav（复杂长语音样本）": "test3.wav",
}


def _format_metric(metrics, prefix):
    """把评估字典的某一类指标整理成展示行。"""
    mse = metrics[f"{prefix}_mse"]
    corr = metrics[f"{prefix}_corr"]
    return mse, corr


def main():
    st.set_page_config(page_title=f"{APP_NAME} v{__version__}",
                       page_icon="🎙️", layout="wide")
    st.title(f"🎙️ {APP_NAME}")
    st.caption(f"哈尔滨工业大学（威海）大一年度项目 · 结题完整版 v{__version__} · "
               f"智能模拟目标 MSE≤0.03 / 相关系数≥0.85")

    # ============ 侧边栏参数 ============
    st.sidebar.header("⚙️ 预处理参数（可自定义）")
    sr = st.sidebar.selectbox("采样率 (Hz)", [16000, 22050, 44100], index=0)
    frame_ms = st.sidebar.slider("帧长 (ms)", 20.0, 30.0, 25.0, 1.0)
    shift_ms = st.sidebar.slider("帧移 (ms)", 10.0, 15.0, 10.0, 1.0)
    window_type = st.sidebar.selectbox("窗函数", ["hamming", "hann", "rect"],
                                       index=0,
                                       format_func=lambda w: {
                                           "hamming": "汉明窗 Hamming",
                                           "hann": "汉宁窗 Hann",
                                           "rect": "矩形窗（不加窗）"}[w])
    coeff = st.sidebar.slider("预加重系数 α", 0.90, 1.0, 0.97, 0.01)
    denoise = st.sidebar.checkbox("启用谱减法降噪", value=False,
                                  help="适合含噪环境录音，如 test2.wav")
    use_vad = st.sidebar.checkbox("启用端点检测", value=True)
    seed = st.sidebar.number_input("随机种子（可复现）", 0, 99999, 42)
    mlp_noise = st.sidebar.slider("MLP 生成扰动强度", 0.0, 1.0, 0.5, 0.05,
                                  help="越大则模拟序列多样性越强，指标略降")

    st.sidebar.markdown("---")
    st.sidebar.header("🎧 选择音频")
    choice = st.sidebar.radio("音频来源", ["上传本地音频", "使用示例音频"])
    audio_path = None
    upload = None
    if choice == "上传本地音频":
        upload = st.sidebar.file_uploader(
            "上传 WAV / MP3 / FLAC 文件",
            type=["wav", "mp3", "flac", "m4a", "ogg"])
    else:
        example = st.sidebar.selectbox("示例音频", list(EXAMPLES.keys()), index=0)
        audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  EXAMPLES[example])

    run = st.sidebar.button("🚀 运行完整流程", type="primary", width="stretch")

    # ============ 主区域 ============
    if not run:
        st.info("在左侧选择音频并调整参数后，点击「🚀 运行完整流程」。")
        st.markdown("""
        **系统能力一览**
        | 模块 | 说明 |
        | --- | --- |
        | 预处理 | 预加重、分帧（帧长/帧移可调）、汉明/汉宁窗、可选谱减法降噪、端点检测 |
        | 特征提取 | 短时能量、短时过零率、短时自相关、基音检测（双阈值） |
        | 统计分析 | 均值/标准差/极值 + 正态/伽马/指数分布拟合 |
        | 参数化模拟 | 移动平均趋势 + 分布噪声，中期指标 MSE≤0.05、r≥0.8 |
        | 智能模拟 | 轻量级 MLP 条件生成（纯 NumPy），结题指标 MSE≤0.03、r≥0.85 |
        | 评估与导出 | MSE/相关系数/达标判定、PNG 图、CSV 数据、TXT 报告 |
        """)
        st.stop()

    if audio_path is None and upload is None:
        st.warning("请上传音频文件或选择示例音频。")
        st.stop()

    # 保存上传文件到临时位置
    temp_path = None
    if upload is not None:
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "output", "_upload")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, upload.name)
        with open(temp_path, "wb") as f:
            f.write(upload.getbuffer())
        audio_path = temp_path

    params = {
        "sr": sr,
        "frame_len_ms": frame_ms,
        "frame_shift_ms": shift_ms,
        "window": window_type,
        "pre_emphasis": coeff,
        "denoise": denoise,
        "vad": use_vad,
        "seed": seed,
        "mlp_noise": mlp_noise,
        "export": False,          # 界面内先不导出，统一由下载按钮提供
        "show_plot": False,
    }

    with st.spinner("正在处理，智能模拟（MLP）训练中……"):
        try:
            r = process_audio(audio_path, params, verbose=False)
        except Exception as e:
            st.error(f"处理失败：{e}")
            st.stop()

    # ============ 结果展示 ============
    m = r["metrics"]
    st.subheader(f"📄 音频信息：{os.path.basename(audio_path)}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("采样率", f"{r['sr']} Hz")
    col2.metric("时长", f"{r['duration_s']:.2f} s")
    col3.metric("帧数", f"{r['n_frames']}")
    col4.metric("处理耗时", f"{r['elapsed_s']:.2f} s")

    st.subheader("✅ 结题指标判定（智能模拟）")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("能量 MSE", f"{m['smart_energy_mse']:.4f}",
              delta="≤0.03 达标" if m["final_pass_energy"] else "未达标",
              delta_color="normal")
    c2.metric("能量 相关系数", f"{m['smart_energy_corr']:.4f}",
              delta="≥0.85 达标" if m["final_pass_energy"] else "未达标",
              delta_color="normal")
    c3.metric("过零率 MSE", f"{m['smart_zcr_mse']:.4f}",
              delta="≤0.03 达标" if m["final_pass_zcr"] else "未达标",
              delta_color="normal")
    c4.metric("过零率 相关系数", f"{m['smart_zcr_corr']:.4f}",
              delta="≥0.85 达标" if m["final_pass_zcr"] else "未达标",
              delta_color="normal")
    c5.metric("总体", "全部达标 ✔" if m["final_pass_all"] else "未达标")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 可视化", "📈 统计分析", "📋 模拟评估", "⬇️ 数据导出"])

    with tab1:
        st.pyplot(r["fig"])

    with tab2:
        st.markdown("**短时能量统计**")
        e = r["stats"]
        st.dataframe({
            "指标": ["样本数", "均值", "标准差", "最小值", "最大值", "最佳拟合分布"],
            "短时能量": [str(e["energy_count"]), f"{e['energy_mean']:.6f}",
                        f"{e['energy_std']:.6f}", f"{e['energy_min']:.6f}",
                        f"{e['energy_max']:.6f}", e["energy_dist_str"]],
            "短时过零率": [str(e["zcr_count"]), f"{e['zcr_mean']:.6f}",
                          f"{e['zcr_std']:.6f}", f"{e['zcr_min']:.6f}",
                          f"{e['zcr_max']:.6f}", e["zcr_dist_str"]],
        }, width="stretch")
        st.markdown("**端点检测与基音检测**")
        if r["vad"]["speech_mask"] is not None:
            st.write(f"检出语音段数：{r['vad']['n_segments']}，"
                     f"语音占比 {r['vad']['speech_ratio'] * 100:.1f}%")
            for i, (s, e2) in enumerate(r["vad"]["boundaries"][:8], 1):
                st.write(f"段 {i}: 帧 {s}~{e2} "
                         f"({s * r['frame_shift'] / r['sr']:.2f}s ~ "
                         f"{e2 * r['frame_shift'] / r['sr']:.2f}s)")
        if r["pitch"].get("f0_hz", 0) > 0:
            st.write(f"基音频率：{r['pitch']['f0_hz']:.1f} Hz"
                     f"（周期 {r['pitch']['lag']} 点，"
                     f"归一化峰值 {r['pitch']['peak']:.4f}）")
        else:
            st.write("未检出明显浊音帧")
        st.markdown("**智能模拟（MLP）训练信息**")
        for tag, info in (("短时能量", r["mlp_info_e"]),
                          ("短时过零率", r["mlp_info_z"])):
            st.write(f"- {tag}：输入 {info['input_dim']} 维 → {info['hidden']} → 1，"
                     f"参数 {info['n_params']} 个，训练 {info['epochs']} 轮，"
                     f"损失 {info['initial_loss']:.4f} → {info['final_loss']:.4f}"
                     f"（下降 {info['loss_improve_pct']:.1f}%）")

    with tab3:
        st.markdown("**结题验收指标：智能模拟 MSE≤0.03、相关系数≥0.85**")
        data = {
            "项目": ["短时能量（参数化模拟）", "短时能量（智能MLP模拟）",
                     "短时过零率（参数化模拟）", "短时过零率（智能MLP模拟）"],
            "MSE": [f"{m['param_energy_mse']:.6f}", f"{m['smart_energy_mse']:.6f}",
                    f"{m['param_zcr_mse']:.6f}", f"{m['smart_zcr_mse']:.6f}"],
            "相关系数": [f"{m['param_energy_corr']:.4f}", f"{m['smart_energy_corr']:.4f}",
                         f"{m['param_zcr_corr']:.4f}", f"{m['smart_zcr_corr']:.4f}"],
            "判定": ["中期达标" if m["mid_pass_energy"] else "未达标",
                     "结题达标" if m["final_pass_energy"] else "未达标",
                     "中期达标" if m["mid_pass_zcr"] else "未达标",
                     "结题达标" if m["final_pass_zcr"] else "未达标"],
        }
        st.dataframe(data, width="stretch")
        st.markdown("**特征有效性自检**")
        sc = r["self_check"]
        st.write(f"综合有效率 {sc['valid_ratio'] * 100:.2f}%"
                 f"（{'达标 ≥95%' if sc['valid_ratio'] >= 0.95 else '未达标'}）")
        for k, v in sc["checks"].items():
            st.write(f"- {k}: {v * 100:.2f}%")

    with tab4:
        st.markdown("点击下方按钮下载导出文件（导出内容基于本次运行结果）：")
        # PNG
        buf = io.BytesIO()
        r["fig"].savefig(buf, dpi=150, bbox_inches="tight")
        buf.seek(0)
        st.download_button("📷 下载结果图 PNG", data=buf,
                           file_name="结果图.png", mime="image/png")

        # CSV
        import csv
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["帧序号", "真实能量", "参数化模拟能量", "智能模拟能量",
                         "真实过零率", "参数化模拟过零率", "智能模拟过零率"])
        for i in range(len(r["energies"])):
            writer.writerow([i + 1, *[f"{v:.8f}" for v in (
                r["energies"][i], r["sim_param_e"][i], r["sim_smart_e"][i],
                r["zcrs"][i], r["sim_param_z"][i], r["sim_smart_z"][i])]])
        st.download_button("📊 下载特征数据 CSV", data=csv_buf.getvalue().encode("utf-8-sig"),
                           file_name="特征数据.csv", mime="text/csv")

        # TXT（复用控制台报告内容，简版）
        txt = f"""语音短时时域特征模拟系统（结题版 v{__version__}）
音频: {os.path.basename(audio_path)}
采样率: {r['sr']} Hz  时长: {r['duration_s']:.2f} s  帧数: {r['n_frames']}

【评估结果（智能模拟）】
短时能量   MSE = {m['smart_energy_mse']:.6f}   相关系数 = {m['smart_energy_corr']:.4f}
短时过零率 MSE = {m['smart_zcr_mse']:.6f}   相关系数 = {m['smart_zcr_corr']:.4f}
结题指标判定: {'全部达标 ✔' if m['final_pass_all'] else '未达标'}

【参数化模拟（中期指标参考）】
短时能量   MSE = {m['param_energy_mse']:.6f}   相关系数 = {m['param_energy_corr']:.4f}
短时过零率 MSE = {m['param_zcr_mse']:.6f}   相关系数 = {m['param_zcr_corr']:.4f}
"""
        st.download_button("📄 下载评估报告 TXT", data=txt.encode("utf-8-sig"),
                           file_name="评估报告.txt", mime="text/plain")


if __name__ == "__main__":
    main()