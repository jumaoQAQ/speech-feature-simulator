# -*- coding: utf-8 -*-
"""
语音短时时域特征模拟系统 - 结题完整版（控制台入口）
====================================================
功能：加载音频 -> 预处理（降噪/预加重/分帧/加窗，参数可自定义）-> 时域特征提取
      （短时能量/过零率/自相关/基音检测）-> 统计分析 -> 参数化模拟 ->
      智能模拟（轻量级 MLP 条件生成）-> 评估（结题指标 MSE≤0.03、r≥0.85）->
      单窗口可视化并导出 PNG / CSV / TXT。

用法：
    python main.py                              # 交互式：输入或拖拽 WAV 路径
    python main.py test1.wav                    # 直接处理指定文件
    python main.py test1.wav --frame-ms 30 --shift-ms 15 --window hann
    python main.py test1.wav --denoise --no-show
    python main.py --list-tests                 # 列出本目录测试音频

结题指标：智能模拟 MSE≤0.03、皮尔逊相关系数≥0.85（详见 vtsim/evaluate.py）。
"""

import os
import sys
import argparse
import glob

# 保证从任意目录运行都可以导入 vtsim 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vtsim.pipeline import process_audio, DEFAULT_PARAMS  # noqa: E402
from vtsim import APP_NAME, __version__                   # noqa: E402


TEST_AUDIO_PATTERNS = ["*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg"]


def list_test_audios(folder):
    """查找目录下的测试音频文件。"""
    files = []
    for pat in TEST_AUDIO_PATTERNS:
        files.extend(glob.glob(os.path.join(folder, pat)))
    return sorted({os.path.abspath(f) for f in files})


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME}（结题完整版 v{__version__}）")
    parser.add_argument("audio", nargs="?", default=None,
                        help="WAV/MP3 音频文件路径（缺省时进入交互模式）")
    parser.add_argument("--list-tests", action="store_true",
                        help="列出本目录及 output 中的测试音频")
    parser.add_argument("--sr", type=int, default=DEFAULT_PARAMS["sr"],
                        help="采样率（默认 16000）")
    parser.add_argument("--frame-ms", type=float,
                        default=DEFAULT_PARAMS["frame_len_ms"],
                        help="帧长 ms（20~30，默认 25）")
    parser.add_argument("--shift-ms", type=float,
                        default=DEFAULT_PARAMS["frame_shift_ms"],
                        help="帧移 ms（10~15，默认 10）")
    parser.add_argument("--window", default=DEFAULT_PARAMS["window"],
                        choices=["hamming", "hann", "rect"],
                        help="窗函数（默认 hamming）")
    parser.add_argument("--coeff", type=float,
                        default=DEFAULT_PARAMS["pre_emphasis"],
                        help="预加重系数（默认 0.97）")
    parser.add_argument("--denoise", action="store_true",
                        help="启用谱减法降噪（默认关闭）")
    parser.add_argument("--no-vad", action="store_true",
                        help="关闭端点检测")
    parser.add_argument("--seed", type=int, default=DEFAULT_PARAMS["seed"],
                        help="随机种子（默认 42，可复现）")
    parser.add_argument("--mlp-noise", type=float,
                        default=DEFAULT_PARAMS["mlp_noise"],
                        help="MLP 生成扰动强度（0~1，默认 0.5）")
    parser.add_argument("--outdir", default=DEFAULT_PARAMS["outdir"],
                        help="输出目录（默认 ./output）")
    parser.add_argument("--no-export", action="store_true",
                        help="不导出 PNG/CSV/TXT")
    parser.add_argument("--show", action="store_true",
                        help="处理完成后弹出图形窗口")
    return parser


def interactive_select(folder):
    """交互模式：列出测试音频并让用户输入路径（支持拖拽）。"""
    print()
    print("=" * 68)
    print(f" {APP_NAME}（结题完整版 v{__version__}）")
    print("=" * 68)
    tests = list_test_audios(folder)
    if tests:
        print("\n可选的测试音频：")
        for i, f in enumerate(tests, 1):
            print(f"  [{i}] {os.path.basename(f)}")
    print("\n请输入 WAV/MP3 音频文件路径（可将文件拖拽到本窗口后回车）：")
    raw = input("文件路径: ").strip().strip('"').strip("'")
    if not raw:
        return None
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(tests):
            return tests[idx - 1]
        print(f"序号 {idx} 超出范围")
        return None
    if not os.path.isabs(raw):
        candidate = os.path.join(folder, raw)
        if os.path.exists(candidate):
            return candidate
    return raw


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    folder = os.path.dirname(os.path.abspath(__file__))

    if args.list_tests:
        tests = list_test_audios(folder)
        print("\n本目录测试音频：")
        for f in tests:
            print("  " + os.path.basename(f))
        print()
        return 0

    audio_path = args.audio
    if audio_path and not os.path.exists(audio_path):
        # 尝试相对本程序目录解析
        candidate = os.path.join(folder, audio_path)
        if os.path.exists(candidate):
            audio_path = candidate
        else:
            print(f"\n错误: 文件不存在 - {audio_path}")
            return 1

    if not audio_path:
        audio_path = interactive_select(folder)
        if not audio_path:
            print("\n未输入文件，退出。")
            return 0
        if not os.path.exists(audio_path):
            print(f"\n错误: 文件不存在 - {audio_path}")
            return 1

    params = {
        "sr": args.sr,
        "frame_len_ms": args.frame_ms,
        "frame_shift_ms": args.shift_ms,
        "window": args.window,
        "pre_emphasis": args.coeff,
        "denoise": args.denoise,
        "vad": not args.no_vad,
        "seed": args.seed,
        "mlp_noise": args.mlp_noise,
        "export": not args.no_export,
        "outdir": args.outdir,
        "show_plot": args.show,
    }

    print(f"\n正在处理: {os.path.basename(audio_path)}")
    print("-" * 68)

    try:
        result = process_audio(audio_path, params, verbose=True)

        r = result
        print()
        print("=" * 68)
        print("处理完成摘要")
        print("=" * 68)
        print(f"采样率: {r['sr']} Hz   时长: {r['duration_s']:.2f} s   "
              f"帧数: {r['n_frames']}   (帧长 {r['frame_len']} 点, "
              f"帧移 {r['frame_shift']} 点)")
        f0 = r["pitch"].get("f0_hz", 0)
        if f0 > 0:
            print(f"基音频率: {f0:.1f} Hz")
        else:
            print("基音检测: 未检出明显浊音帧")
        if r["vad"]["speech_mask"] is not None:
            print(f"端点检测: {r['vad']['n_segments']} 个语音段, "
                  f"语音占比 {r['vad']['speech_ratio'] * 100:.1f}%")

        m = r["metrics"]
        print()
        print("【评估结果】")
        print(f"  参数化模拟   能量: MSE={m['param_energy_mse']:.6f} r={m['param_energy_corr']:.4f} | "
              f"过零率: MSE={m['param_zcr_mse']:.6f} r={m['param_zcr_corr']:.4f}")
        print(f"  智能MLP模拟  能量: MSE={m['smart_energy_mse']:.6f} r={m['smart_energy_corr']:.4f} | "
              f"过零率: MSE={m['smart_zcr_mse']:.6f} r={m['smart_zcr_corr']:.4f}")

        sc = r["self_check"]
        print(f"  特征有效性自检: {sc['valid_ratio'] * 100:.2f}% "
              f"({'达标' if sc['valid_ratio'] >= 0.95 else '未达标'})")

        if r["metrics"]["final_pass_all"]:
            print("\n  >>> 结题指标判定: 全部达标（智能模拟 MSE≤0.03, 相关系数≥0.85）")
        else:
            print("\n  >>> 结题指标判定: 尚未全部达标（详见导出的报告）")

        if r["files"]:
            print("\n导出文件：")
            for kind, path in r["files"].items():
                print(f"  {kind.upper():>4}: {path}")
        print(f"\n处理耗时: {r['elapsed_s']:.2f} 秒")

    except Exception as e:
        print(f"\n处理失败: {e}")
        import traceback
        traceback.print_exc()
        try:
            input("\n按回车键退出...")
        except EOFError:
            pass
        return 1

    try:
        input("\n按回车键退出...")
    except EOFError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())