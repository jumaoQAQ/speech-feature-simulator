# -*- coding: utf-8 -*-
"""
语音短时时域特征模拟系统 (Voice Time-domain Simulation System) - 核心算法包
=====================================================================
模块划分（对应立项报告中的“模块化设计”）：
    preprocess  预处理模块：音频加载、预加重、分帧、加窗、谱减法降噪、端点检测
    features    特征提取模块：短时能量、短时过零率、短时自相关、基音检测
    statistics  统计分析模块：特征统计量、概率分布拟合
    simulate    参数化模拟模块：分布采样 + 时序平滑
    neural_sim  智能模拟模块：轻量级 MLP 神经网络条件生成（纯 NumPy 实现）
    evaluate    评估模块：MSE / 相关系数 / 结题指标判定
    report      可视化与导出模块：PNG 结果图、CSV 数据、TXT 报告
    pipeline    主流程：串联上述全部模块

本系统按“大一年度项目计划”结题指标实现：
    智能模拟：MSE <= 0.03，皮尔逊相关系数 >= 0.85
    支持参数自定义、降噪、端点检测、特征曲线可视化与 CSV/PNG 数据导出。
"""

__version__ = "2.0.0"
APP_NAME = "语音短时时域特征模拟系统"

from . import preprocess, features, statistics, simulate, neural_sim, evaluate, report, pipeline  # noqa: E402,F401