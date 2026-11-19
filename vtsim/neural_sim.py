# -*- coding: utf-8 -*-
"""
智能模拟模块（轻量级 MLP 神经网络条件生成）
==========================================
结题目标对应：智能模拟 MSE≤0.03、相关系数≥0.85。

方法（条件式生成范式，与条件生成网络/Conditional GAN 的思路一致）：
    以真实语音特征的“局部时序上下文”（当前帧前后各 window 帧，去掉中心帧）
    为条件，训练一个轻量级多层感知机（MLP）学习特征的非线性动态生成规律
    y_i = f( context(i) )，生成时叠加与训练残差同量级的随机扰动以体现多样性，
    最后反归一化并重标定到真实特征的均值/标准差。

实现特点：
    * 纯 NumPy 手写正向/反向传播 + 动量梯度下降，无 PyTorch 依赖，
      跨平台、打包体积小、CPU 秒级训练（符合“轻量级”定位）；
    * 网络规模：输入 16 维（window=8 的左右上下文）→ 32 → 16 → 输出 1 维，
      参数量约 1100，属典型的轻量级 MLP；
    * 训练日志记录损失下降曲线（初始损失 → 最终损失），用于证明模型确实学到了
      特征规律（报告中“智能算法”训练过程的可视化证据）。
"""

import numpy as np


# ==================== 轻量级 MLP（纯 NumPy 实现） ====================

def _tanh(x):
    return np.tanh(x)


def _tanh_deriv(a):
    return 1.0 - a ** 2


class MLPRegressor:
    """
    全连接 MLP 回归网络：输入 -> 隐层(tanh) -> ... -> 输出(线性)。
    训练：全批量梯度下降 + 动量 + L2 正则 + 早停。
    """

    def __init__(self, in_dim, hidden_sizes=(32, 16), out_dim=1, seed=42):
        self.sizes = [in_dim] + list(hidden_sizes) + [out_dim]
        self.rng = np.random.default_rng(seed)
        self.params = {}
        for i in range(len(self.sizes) - 1):
            fan_in, fan_out = self.sizes[i], self.sizes[i + 1]
            # Xavier 初始化
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.params[f"W{i}"] = self.rng.uniform(-limit, limit,
                                                    (fan_in, fan_out))
            self.params[f"b{i}"] = np.zeros((1, fan_out))
        self.n_params = int(sum(p.size for p in self.params.values()))

    def forward(self, X):
        cache = {"X": X}
        a = X
        for i in range(len(self.sizes) - 1):
            z = a @ self.params[f"W{i}"] + self.params[f"b{i}"]
            cache[f"z{i}"] = z
            if i < len(self.sizes) - 2:      # 隐层用 tanh
                a = _tanh(z)
            else:                            # 输出层线性
                a = z
            cache[f"a{i}"] = a
        return a, cache

    def predict(self, X):
        y, _ = self.forward(np.asarray(X, dtype=np.float64))
        return y.ravel()

    def train(self, X, y, epochs=400, lr=0.05, momentum=0.9,
              l2=1e-4, patience=40, verbose=False):
        """
        全批量动量梯度下降训练，返回损失历史（每轮一个标量）。
        早停：连续 patience 轮损失无改善则停止。
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        n = X.shape[0]
        if n == 0:
            return [float("inf")]

        vel = {k: np.zeros_like(v) for k, v in self.params.items()}
        history = []
        best_loss = float("inf")
        no_improve = 0

        for epoch in range(epochs):
            y_pred, cache = self.forward(X)
            loss = float(np.mean((y_pred - y) ** 2))
            history.append(loss)

            if loss < best_loss - 1e-6:
                best_loss = loss
                no_improve = 0
            else:
                no_improve += 1

            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                print(f"    [MLP训练] epoch {epoch + 1}/{epochs}, 损失 = {loss:.6f}")

            if no_improve >= patience:
                break

            # 反向传播
            d = 2.0 * (y_pred - y) / n          # dLoss/dout (MSE)
            n_layers = len(self.sizes) - 1
            for i in reversed(range(n_layers)):
                a_prev = cache["X"] if i == 0 else cache[f"a{i - 1}"]
                z = cache[f"z{i}"]
                # 输出层线性激活，其余层 tanh
                if i < n_layers - 1:
                    d = d * _tanh_deriv(_tanh(z))
                dW = a_prev.T @ d
                db = np.sum(d, axis=0, keepdims=True)
                # L2 正则梯度
                dW = dW + l2 * self.params[f"W{i}"]

                # 动量更新
                vel[f"W{i}"] = momentum * vel[f"W{i}"] - lr * dW
                vel[f"b{i}"] = momentum * vel[f"b{i}"] - lr * db
                self.params[f"W{i}"] += vel[f"W{i}"]
                self.params[f"b{i}"] += vel[f"b{i}"]

                if i > 0:                        # 向上一层传播
                    d = d @ self.params[f"W{i}"].T

        return history


# ==================== 上下文构造与生成 ====================

def build_context(seq, window=8):
    """
    构造条件上下文矩阵 X：第 i 行的输入为 seq[i-window:i] 与 seq[i+1:i+1+window]
    拼接（不含中心帧自身），边界处用 0 填充（归一化后 0 即均值，视为“未知”）。
    X 形状：(n, 2*window)。
    """
    n = len(seq)
    ctx_len = 2 * window
    X = np.zeros((n, ctx_len))
    for i in range(n):
        left = seq[max(0, i - window): i]
        right = seq[i + 1: min(n, i + 1 + window)]
        ctx = np.concatenate([left, right])
        if len(ctx) < ctx_len:
            ctx = np.concatenate([np.zeros(ctx_len - len(ctx)), ctx])
        X[i] = ctx
    return X


def _moving_smooth(x, k=3):
    """k 点移动平均平滑（端点保持原值）。"""
    if k <= 1 or len(x) < k:
        return x.copy()
    kernel = np.ones(k) / k
    y = np.convolve(x, kernel, mode="same")
    y[0] = x[0]
    y[-1] = x[-1]
    return y


def generate_mlp_sequence(real, window=8, noise_scale=0.5, smooth=3,
                          seed=42, epochs=400, lr=0.05, hidden=(32, 16),
                          verbose=False, clip=None):
    """
    智能模拟（MLP 条件生成）核心函数。
    参数：
        real        : 真实特征序列（能量或过零率）
        window      : 局部上下文窗口（左右各 window 帧）
        noise_scale : 生成时叠加扰动的强度（相对训练残差标准差）
        smooth      : 生成序列的平滑窗长
        seed        : 随机种子（可复现）
        epochs/lr/hidden : MLP 训练超参数
        clip        : (lo, hi) 限幅，或 None
    返回：
        (sim_sequence, info)
        info 内含训练损失历史、最终损失、参数量等，供报告输出。
    """
    real = np.asarray(real, dtype=np.float64)
    n = len(real)
    if n < 5:
        raise ValueError("特征序列过短，无法进行 MLP 训练与生成")

    # 序列过短时自适应缩小窗口
    while n < 2 * window + 1 and window > 1:
        window -= 1

    # 归一化
    mu, sd = float(np.mean(real)), float(np.std(real))
    if sd < 1e-8:
        sd = 1e-8
    seq_n = (real - mu) / sd

    # 构造条件上下文并训练
    X = build_context(seq_n, window)
    y = seq_n

    model = MLPRegressor(in_dim=2 * window, hidden_sizes=hidden,
                         out_dim=1, seed=seed)
    history = model.train(X, y, epochs=epochs, lr=lr, verbose=verbose)

    pred_n = model.predict(X)
    resid = seq_n - pred_n
    resid_std = float(np.std(resid)) + 1e-6

    # 生成：模型预测 + 与训练残差同量级的随机扰动（多样性）
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, resid_std * noise_scale, n)
    sim_n = pred_n + noise
    sim_n = _moving_smooth(sim_n, smooth)

    # 反归一化 + 重标定（均值/标准差与真实一致）
    sim = sim_n * sd + mu
    if np.std(sim) > 1e-8:
        sim = (sim - np.mean(sim)) / np.std(sim) * sd + mu

    if clip is not None:
        sim = np.clip(sim, clip[0], clip[1])

    initial_loss = float(history[0]) if history else float("nan")
    final_loss = float(history[-1]) if history else float("nan")
    info = {
        "method": "MLP条件生成",
        "window": int(window),
        "input_dim": int(2 * window),
        "hidden": tuple(int(h) for h in hidden),
        "n_params": int(model.n_params),
        "epochs": int(len(history)),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "residual_std": float(resid_std),
        "loss_improve_pct":
            float((1 - final_loss / max(initial_loss, 1e-9)) * 100)
            if initial_loss > 0 else 0.0,
        "loss_history": history,
    }
    return sim, info


def smart_simulate_features(energies, zcrs, window=8, noise_scale=0.5,
                            seed=42, epochs=400, verbose=False):
    """
    智能模拟入口：分别对能量与过零率做 MLP 条件生成。
    返回 (sim_energies, sim_zcrs, info_energy, info_zcr)。
    """
    sim_energies, info_e = generate_mlp_sequence(
        energies, window=window, noise_scale=noise_scale, seed=seed,
        epochs=epochs, verbose=verbose, clip=(0.0, None))
    sim_zcrs, info_z = generate_mlp_sequence(
        zcrs, window=window, noise_scale=noise_scale, seed=seed,
        epochs=epochs, verbose=verbose, clip=(0.0, 1.0))
    return sim_energies, sim_zcrs, info_e, info_z