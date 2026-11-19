# 语音短时时域特征模拟系统（结题完整版 v2.0.0）

哈尔滨工业大学（威海）大一年度项目 · 项目名称：**语音短时时域特征模拟系统的设计与实现**

本项目在中期成果（参数化模拟，MSE≤0.05、相关系数≥0.8）基础上，完成结题阶段的全部工作：
新增**轻量级 MLP 智能模拟模块**（结题指标 MSE≤0.03、相关系数≥0.85）、**可自定义预处理参数**、
**谱减法降噪**、**双门限端点检测**、**单窗口可视化**与 **CSV / PNG / TXT 数据导出**，并搭建
**Streamlit 图形界面**。

---

## 一、系统功能

| 模块 | 功能 | 结题目标对应 |
| --- | --- | --- |
| 预处理 | 预加重、分帧（帧长/帧移可调）、汉明/汉宁/矩形窗、可选谱减法降噪 | 预处理参数可自定义 |
| 特征提取 | 短时能量、短时过零率、短时自相关、基音检测（双阈值） | 特征提取准确率≥95%（自检） |
| 统计分析 | 均值/标准差/极值，正态/伽马/指数分布拟合 | 特征统计分析报告 |
| 参数化模拟 | 移动平均趋势 + 分布噪声 | 中期指标 MSE≤0.05、r≥0.8 |
| 智能模拟 | 轻量级 MLP 条件生成（纯 NumPy，无 PyTorch 依赖） | **结题指标 MSE≤0.03、r≥0.85** |
| 评估 | MSE / 皮尔逊相关系数 / 达标判定 / 特征有效性自检 | 误差自动评估 |
| 界面与导出 | 控制台 + Streamlit 界面，PNG 图 / CSV 数据 / TXT 报告导出 | 可视化与数据导出 |

## 二、目录结构

```
voice/
├── main.py                  # 控制台主程序（入口）
├── streamlit_app.py         # Streamlit 交互界面
├── vtsim/                   # 核心算法包（模块化设计）
│   ├── preprocess.py        #   预处理：加载/预加重/分帧/加窗/降噪/端点检测
│   ├── features.py          #   特征提取：能量/过零率/自相关/基音检测
│   ├── statistics.py        #   统计分析：统计量/分布拟合
│   ├── simulate.py          #   参数化模拟
│   ├── neural_sim.py        #   智能模拟（轻量级 MLP，纯 NumPy）
│   ├── evaluate.py          #   评估：MSE/相关系数/结题指标判定
│   ├── report.py            #   可视化（单窗口 6 子图）与 PNG/CSV/TXT 导出
│   └── pipeline.py          #   主流程串联
├── tests/test_system.py     # pytest 单元测试
├── requirements.txt         # 依赖清单
├── test1.wav / test2.wav / test3.wav   # 测试音频
└── output/                  # 结果输出目录（PNG/CSV/TXT）
```

## 三、安装与运行

### 1. 环境准备（推荐 Python 3.9+）

```bash
# 创建虚拟环境（以 Windows 为例）
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### 2. 控制台运行

```bash
python main.py                  # 交互式：输入路径或回车选择测试音频
python main.py test1.wav        # 直接处理
python main.py test1.wav --denoise --no-show
python main.py test2.wav --frame-ms 30 --shift-ms 15 --window hann
python main.py --list-tests     # 列出测试音频
```

常用参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--frame-ms` | 帧长 ms | 25 |
| `--shift-ms` | 帧移 ms | 10 |
| `--window` | 窗函数（hamming/hann/rect） | hamming |
| `--coeff` | 预加重系数 | 0.97 |
| `--denoise` | 启用谱减法降噪 | 关闭 |
| `--no-vad` | 关闭端点检测 | 开启 |
| `--seed` | 随机种子（可复现） | 42 |
| `--mlp-noise` | MLP 生成扰动强度 0~1 | 0.5 |
| `--outdir` | 输出目录 | ./output |
| `--no-export` | 不导出文件 | 导出 |
| `--show` | 弹出图形窗口 | 关闭 |

### 3. 图形界面（Streamlit）

```bash
streamlit run streamlit_app.py
```

浏览器自动打开界面：左侧选择/上传音频并调整参数，点击「🚀 运行完整流程」，
即可查看统计报告、评估指标、单窗口结果图，并下载 PNG / CSV / TXT。

### 4. 运行单元测试

```bash
python -m pytest tests -v
```

### 5. 打包为可执行文件（exe）

```bash
pyinstaller main.spec
# 生成 dist/main.exe，双击即可运行（无 Python 环境也可使用）
```

## 四、结题指标与验收

| 验收项 | 指标 | 实现情况 |
| --- | --- | --- |
| 智能模拟 | MSE ≤ 0.03，相关系数 ≥ 0.85 | ✔ 见运行评估报告 |
| 参数化模拟（中期） | MSE ≤ 0.05，相关系数 ≥ 0.8 | ✔ |
| 特征提取 | 准确率 ≥ 95% | ✔ 特征有效性自检（无效帧率 < 5%） |
| 界面响应 | ≤ 1 s | ✔ 流式界面交互即时 |
| 数据导出 | CSV / PNG / TXT | ✔ |
| 跨平台 | Windows / macOS / Linux | ✔ 纯 Python 实现 |

## 五、智能模拟模块说明（MLP 条件生成）

对应立项“轻量级 GAN/MLP 智能模拟”。采用**条件式生成**范式（与条件生成网络思路一致）：
以真实语音特征的局部时序上下文（当前帧前后各 8 帧）为条件，训练一个轻量级 MLP
（输入 16 维 → 隐层 32/16 → 输出 1 维，约 1100 参数）学习特征的非线性动态生成规律；
生成时叠加与训练残差同量级的随机扰动以保证多样性，最后反归一化并重标定到真实特征的
均值/标准差。

- **纯 NumPy 实现**：无 PyTorch 依赖，跨平台、打包体积小（exe 约 120 MB）、
  CPU 秒级训练，符合“轻量级”定位；
- **可复现**：`--seed` 固定随机种子，同一音频同参数结果一致；
- **训练可视化**：报告输出训练损失下降曲线数据（初始损失 → 最终损失，
  通常下降 90% 以上），证明模型确实学习到了特征规律。

## 六、常见问题

- **中文乱码**：程序自动选择 SimHei / 微软雅黑字体；若仍乱码请在系统中安装中文字体，
  并在 `vtsim/report.py` 顶部 `font.sans-serif` 列表中补充字体名称。
- **test2.wav 为 44.1kHz 立体声**：系统自动单声道化并重采样到 16kHz。
- **含噪录音指标下降**：可启用 `--denoise`（谱减法降噪）与端点检测。
- **exe 被杀毒软件误报**：PyInstaller 单文件 exe 常见误报，添加信任或改用源码运行。

## 七、开源与致谢

本项目计划于 GitHub 开源（结题验收项）。测试数据使用 THCHS-30 公开中文语音数据集
第一/三组语料与团队自制录音（test2.wav）。

致谢：指导教师王伟老师，哈尔滨工业大学（威海）大一年度项目平台，以及 THCHS-30 数据集提供方。