---
layout:     post
title:      优化器 (3)
subtitle:   Gram Newton-Schulz
paper_url:  http://dao-lab.ai/blog/2026/gram-newton-schulz/
github_repo: https://github.com/Dao-AILab/gram-newton-schulz/
date:       2026-04-05
author:     Zifeng Mai
header-img: img/fuji1.jpg
preview:    本文介绍了 Gram Newton-Schulz 算法，这是 Muon 优化器中 Newton-Schulz 正交化过程的高效变体。通过在 Gram 矩阵上迭代而非原矩阵，Gram Newton-Schulz 显著减少了 FLOPs 和运行时间。本文详细推导了算法的数学原理，分析了数值稳定性问题，并给出了完整的理论证明。
catalog: true
series: 优化器
tags:
    - Optimizer
    - Muon
    - Newton-Schulz
    - Optimization Theory
    - LLM
---

## 引言

在上一篇文章中，我们深入介绍了 Muon 优化器的数学原理。Muon 的核心创新在于使用 Newton-Schulz 迭代来近似矩阵的 msign 函数，也称为极分解（polar decomposition），从而实现对参数矩阵的正交化更新。

然而，标准的 Newton-Schulz 算法存在一个显著的计算瓶颈：它需要在每次迭代中对原始的矩形参数矩阵进行多次矩阵乘法运算。对于一个 $n \times m$ 的权重矩阵，每次 Newton-Schulz 迭代需要 $O(mn^2)$ 的时间复杂度，这在大规模训练中成为了不可忽视的开销。

近期，DAO Lab 提出了 **Gram Newton-Schulz** 算法，通过以下创新显著加速了 Muon 优化器：

1. 数学等价性：在 Gram 矩阵 $\mathbf{X}\mathbf{X}^\top$ 上迭代而非原矩阵 $\mathbf{X}$，输出与标准 Newton-Schulz 数学等价
2. 数值稳定性：通过重启策略和 float16 算术，实现与标准方法相当的稳定性
3. 硬件感知：对称 GEMM 核充分利用 Hopper/Blackwell 架构特性

在 Kimi K2 这种万亿参数量级别的模型上，Gram Newton-Schulz 可以将优化器时间减少高达 50%。

## 一、Muon 与 Newton-Schulz 回顾

### 1.1. Muon 优化器更新规则

在第 $k$ 步训练时，令 $\mathbf{W}_k \in \mathbb{R}^{n \times m}$ 为权重矩阵，$\mathbf{G}_k$ 是 $\mathbf{W}_k$ 的梯度。Muon 的更新规则为：

$$
\begin{equation}
\begin{aligned}
\mathbf{M}_k &= \mu \mathbf{M}_{k-1} + \mathbf{G}_k \\
\mathbf{W}_{k+1} &= \mathbf{W}_k - \eta \cdot \text{polar}(\mathbf{M}_k)
\end{aligned}
\end{equation}
$$

其中 $\mu$ 是动量系数，$\eta$ 是学习率。$\mathbf{M}_k$ 是动量矩阵，$\mathbf{M}_0 = 0$。

**Definition 1（极分解）**：若 $\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$ 是矩阵 $\mathbf{X}$ 的 SVD 分解，则 $\text{polar}(\mathbf{X}) = \mathbf{U}\mathbf{V}^\top$。

由于精确计算 $\text{polar}(\mathbf{X})$ 需要进行完整的 SVD 分解，计算开销较大，因此 Muon 使用 Newton-Schulz 迭代来近似它。

### 1.2. 标准 Newton-Schulz 迭代

Newton-Schulz 是一种基于矩阵多项式的迭代方法。从初始矩阵 $\mathbf{X}_0$ 开始，每次迭代按照以下规则更新近似值 $\mathbf{X}_t \approx \text{polar}(\mathbf{X})$：

$$
\begin{equation}
\mathbf{X}_{t+1} = a_t \mathbf{X}_t + b_t \mathbf{X}_t \mathbf{X}_t^\top \mathbf{X}_t + c_t \left(\mathbf{X}_t \mathbf{X}_t^\top\right)^2 \mathbf{X}_t
\label{eq:newton-schulz}
\end{equation}
$$

其中 $(a_t, b_t, c_t)$ 是预先设定的多项式系数。

下面分析标准 Newton-Schulz 迭代的计算复杂度。

令 $T$ 表示迭代次数（在 Muon 中 $T=5$），并假设 $n \leq m$，定义纵横比 $\alpha = m/n \geq 1$。

每次迭代包含三次矩阵乘法：
- $\mathbf{X}\mathbf{X}^\top$：$2mn^2$ FLOPs
- $\mathbf{A}^2$（其中 $\mathbf{A} = \mathbf{X}\mathbf{X}^\top$）：$2n^3$ FLOPs
- $\mathbf{B}\mathbf{X}$（其中 $\mathbf{B} = b_t\mathbf{A} + c_t\mathbf{A}^2$）：$2mn^2$ FLOPs

总计算量为：

$$
\begin{equation}
T(4mn^2 + 2n^3) = 2T(2\alpha + 1)n^3 \text{ FLOPs}
\end{equation}
$$

当 $T=5$ 时，总计算量为 $(20\alpha + 10)n^3$，分布在 15 次 GEMM 操作上。

因此，标准的 Newton-Schulz 迭代存在以下问题：

1. **未利用对称性**：$\mathbf{A} = \mathbf{X}\mathbf{X}^\top$ 和 $\mathbf{B}$ 都是对称矩阵，但标准实现未利用这一结构。
2. **强依赖 $\alpha$**：当 $\alpha \gg 1$时，计算量主要由昂贵的 GEMM 操作主导。

## 二、Gram Newton-Schulz 的数学原理

### 2.1. 核心思想

Gram Newton-Schulz 的核心洞察在于下面的公式 \eqref{eq:polar}（证明见上一篇[博客](/2026/01/28/Muon/#apd1-proof-on-eq-eqrefeqmsign)）。

$$
\begin{equation}
\text{polar}(\mathbf{X}) = (\mathbf{X}\mathbf{X}^\top)^{-1/2} \mathbf{X}
\label{eq:polar}
\end{equation}
$$

基于公式 \eqref{eq:polar}，Gram Newton-Schulz 的策略是：
1. 计算 $n \times n$ Gram 矩阵 $\mathbf{X}\mathbf{X}^\top$
2. 使用迭代方法近似 $\mathbf{Q}_T \approx (\mathbf{X}\mathbf{X}^\top)^{-1/2}$
3. 计算 $\mathbf{Q}_T \mathbf{X}$

Gram Newton-Schulz 的关键优势在于，第 2 步（占据绝大部分计算量）仅在小的 $n \times n$ 对称矩阵上操作，仅需两次矩形矩阵乘法（初始的 $\mathbf{X}\mathbf{X}^\top$ 和最终的 $\mathbf{Q}_T\mathbf{X}$）。

### 2.2. 从 Newton-Schulz 到 Gram Newton-Schulz

**Proposition 1（奇异向量不变性）**：设 $\mathbf{X}_0 = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$ 是 $\mathbf{X}_0$ 的 SVD 分解。若 $\mathbf{X}_t$ 按照方程 \eqref{eq:newton-schulz} 迭代，则对任意 $t \geq 0$，存在奇次多项式 $p_t: \mathbb{R} \to \mathbb{R}$ 使得：

$$
\mathbf{X}_t = \mathbf{U} p_t(\mathbf{\Sigma}) \mathbf{V}^\top
$$

其中 $p_t(\mathbf{\Sigma}) = \text{diag}(p_t(\sigma_1), \ldots, p_t(\sigma_r))$，$r$ 为 $\mathbf{X}_0$ 的秩。

根据 Proposition 1，如果多项式序列 $(p_T \circ \cdots \circ p_1)(x) \approx 1$ 对所有奇异值成立，则 $\mathbf{X}_T \approx \mathbf{U}\mathbf{V}^\top = \text{polar}(\mathbf{X}_0)$。

那么，如何将迭代多项式方法 $(p_T \circ \cdots \circ p_1)(\mathbf{X}) \approx \text{polar}(\mathbf{X})$ 转换为近似 $\mathbf{Y} \mapsto \mathbf{Y}^{-1/2}$ 的迭代方法？

**Lemma 1（奇次多项式的性质）** 每个奇次多项式 $p(x) = ax + bx^3 + cx^5$ 可以重写为 $p(x) = x h(x^2)$ 的形式，其中 $h(y) = a + by + cy^2$ 是低一次的多项式。

因此，如果 $p(x) \approx 1$，则 $h(y) = p(y^{1/2})y^{-1/2} \approx y^{-1/2}$，因此 Newton-Schulz 多项式隐式地提供了近似逆平方根的方法。下面的定理展示了二者的对应关系。

**Theorem 1（迭代求逆平方根）**：设任取 $t \in \{1, \ldots, T\}$ 都有

$$
\begin{equation}
    p_t(x) = x h_t(x^2)
\end{equation}
$$

则有

$$
\begin{equation}
    (p_T \circ \cdots \circ p_1)(x) = q_T x
\end{equation}
$$

其中 $q_T$ 由以下迭代定义：

$$
\begin{equation}
\begin{cases}
z_t = h_t(r_{t-1}) \\
r_t = r_{t-1} z_t^2 \\
q_t = q_{t-1} z_t
\end{cases}
\end{equation}
$$

迭代的起始条件为 $r_0 = x^2$，$q_0 = 1$。

注意到当 $t\to+\infty$ 时，有 $q_t = x_t/x_0 \to 1/x_0 = (x_0^2)^{-1/2} = r_0^{-1/2}$。因此，我们就可以通过迭代来近似 $r_0^{-1/2}$。

### 2.3. Naive Gram Newton-Schulz 算法

将定理 1 的迭代提升到矩阵形式，作者提出了 Naive Gram Newton-Schulz 算法：

**算法 1（Naive Gram Newton-Schulz）**

输入：$\mathbf{X} \in \mathbb{R}^{n \times m}$（$n \leq m$），系数 $\{(a_t, b_t, c_t)\}_{t=1}^5$

1. $\mathbf{X} \gets \mathbf{X} / (\Vert\mathbf{X}\Vert_{\mathsf{F}} + \epsilon)$（将奇异值归一化到 $[0, 1]$，$\epsilon = 10^{-7}$）
2. $\mathbf{R}_0 = \mathbf{X} \mathbf{X}^\top$
3. $\mathbf{Q}_0 = \mathbf{I}$
4. 对于 $t = 1, \ldots, 5$ 按照下面的公式迭代
   
   $$
   \begin{equation}
    \begin{aligned}
   \mathbf{Z}_t &\gets a_t\mathbf{I} + b_t \mathbf{R}_{t-1} + c_t \mathbf{R}_{t-1}^2\\
   \mathbf{Q}_t &\gets \mathbf{Q}_{t-1} \mathbf{Z}_t\\
   \mathbf{R}_t &\gets \mathbf{Z}_t \mathbf{R}_{t-1} \mathbf{Z}_t
    \end{aligned}
   \end{equation}
   $$

5. 返回 $\mathbf{Q}_5 \mathbf{X}$

### 2.4. 计算复杂度分析

下面分析 Naive Gram Newton-Schulz 的 FLOPs 数量级。每次迭代包含四次矩阵乘法（使用对称 GEMM 核）：
- $\mathbf{R}_{t-1}^2$：$n^3$ FLOPs
- $\mathbf{Q}_{t-1} \mathbf{Z}_t$：$n^3$ FLOPs
- $\mathbf{Z}_ {t} \mathbf{R}_ {t-1} \mathbf{Z}_ {t}$：$2n^3$ FLOPs

初始化和输出步骤：
- $\mathbf{X}\mathbf{X}^\top$：$mn^2$ FLOPs
- $\mathbf{Q}_5 \mathbf{X}$：$2mn^2$ FLOPs（非对称）

下面的计算量可以优化：
- $\mathbf{Q}_ 1 = \mathbf{Q}_ {0}\mathbf{Z}_ 1=\mathbf{Z}_1$ 不需要计算：节省 $n^3$ FLOPs
- $\mathbf{R}_5$ 不需要计算：节省 $2n^3$ FLOPs

因此 Naive Gram Newton-Schulz 的总 FLOPs 为：

$$
\begin{equation}
T \cdot 4n^3 + 3mn^2 - 3n^3 = (4T + 3\alpha - 3)n^3
\end{equation}
$$

当 $T=5$ 时，总计算量为 $(17 + 3\alpha)n^3$，分布在 19 次 GEMM 操作上。

使用对称 GEMM 的标准 Newton-Schulz 需要 $T(3\alpha + 1)n^3$ FLOPs。当 $\alpha > 1$（非方阵）时，Gram Newton-Schulz 更优。

在 Muon 的场景下（$T=5, \alpha=4$）：
- 相比使用对称 GEMM 的标准 Newton-Schulz 节省 **55%** FLOPs
- 相比典型实现（无对称 GEMM）节省 **68%** FLOPs

## 三、Stabilized Gram Newton-Schulz

### 3.1. Naive Gram Newton-Schulz 的数值稳定性问题s

尽管 Naive Gram Newton-Schulz 在精确算术下与标准 Newton-Schulz 等价，但在有限精度（尤其是半精度）下表现较差。实验发现，使用 bfloat16 训练 Transformer 时损失函数曲线会频繁出现尖峰，最终输出充满 Inf。

作者通过研究特征值和奇异值的演化，给出了矩阵发散的几个原因。

**原因 1：伪负特征值（Spurious Negative Eigenvalues）**

根据定义，$\mathbf{R}_t$ 应该是半正定矩阵（因为 $r_t = x_t^2 \geq 0$）。然而，在 bfloat16 下，$\mathbf{R}_0 = \mathbf{X}_0\mathbf{X}_0^\top$ 会引入微小的负特征值（如 $-10^{-5}$），这是由于浮点误差导致的。下面的命题指出，只要出现了负特征值，该特征值的幅度就会随着迭代而指数增长。

**Proposition 2（负特征值指数增长）**：设 $\mathbf{R}_0$ 为 Gram Newton-Schulz 迭代的初始 Gram 矩阵。若 $\mathbf{R}_0$ 存在负特征值 $\lambda < 0$，则该特征值的幅度随迭代步数指数增长。

**Proof**：回顾更新规则 $r_t = r_{t-1} z_t^2 = r_{t-1} h_t(r_{t-1})^2$。

代入 $h_t(x) = \frac{15}{8} - \frac{10}{8}x + \frac{3}{8}x^2$，当 $r_{t-1} < 0$ 时：

$$
r_t < \left(\frac{15}{8}\right)^2 r_{t-1}
$$

因此，若 $r_0 < 0$，伪特征值的幅度指数增长，引发链式反应：当 $r_t \to -\infty$ 时，$z_t \to \infty$，导致 $q_t \to \infty$ 和 $x_t \to \infty$。$\blacksquare$

**原因 2：特征向量漂移（Eigenvector Drift）**

在精确算术下，所有中间矩阵的特征向量与 $\mathbf{X}_0$ 的左奇异向量 $\mathbf{U}$ 匹配，但在有限精度下会发生漂移。这导致 $\mathbf{Q}_t$ 和 $\mathbf{X}_t$ 的特征值偏离理论值。

### 3.2. 重启策略（Restarting）

为了缓解 Naive Gram Newton-Schulz 在低精度下的数值稳定性问题，作者提出了重启策略。重启的核心思想在于不直接计算 $\mathbf{X}_ T$，而是运行少量迭代（如 5 步）得到 $\mathbf{X}_ 5$，然后在 $\mathbf{X}_ 5$ 上再次应用 Gram Newton-Schulz 计算 $\mathbf{X}_ {10}$，依此类推。

每次重启时，由于重新计算了 Gram 矩阵 $\mathbf{R} = \mathbf{X}\mathbf{X}^\top$，这消除了之前积累的大幅度负特征值。另外，由于 $\mathbf{Q}_t$ 在每次重启时重置为单位矩阵，其特征值始终有界，因此 $\mathbf{X}_t = \mathbf{Q}_t \mathbf{X}$ 的特征值也能保持有界。

作者提出，重启的最佳时机取决于所使用的多项式系数序列。对于 Muon 来说，作者使用 Polar Express 的五次多项式：

| $t$ | $a_t$ | $b_t$ | $c_t$ |
|:---:|:-----:|:-----:|:-----:|
| 1 | 8.123737 | -22.232240 | 16.373715 |
| 2 | 4.026529 | -2.776323 | 0.514551 |
| 3 | 3.870284 | -2.739120 | 0.520999 |
| 4 | 3.253351 | -2.343223 | 0.481420 |
| 5 | 2.300652 | -1.668904 | 0.418807 |

通过数值模拟发现，**在第 2 次迭代后重启**可以最好地平衡稳定性和速度：
- $\mathbf{R}_t$ 的最小特征值保持在 $-0.4$ 以上
- $\mathbf{Q}_t$ 的条件数保持在 $\approx 100$ 以下

### 3.3. 进一步的安全措施

**使用 float16 而非 bfloat16**

float16 的范围（约 $6.1 \times 10^{-5}$ 到 $6.5 \times 10^4$）足够使用，且精度更高。在某些测试矩阵上，float16 给出更准确的 $\text{polar}(\mathbf{X})$ 近似。

**引入安全因子**

大多数 Newton-Schulz 多项式设计为在 $\Vert\mathbf{X}_0\Vert \leq 1$ 时收敛。由于数值误差可能导致奇异值略大于 1，因此作者建议引入安全因子：

$$
\tilde{p}_t(x) = p_t(x / 1.05)
$$

这确保即使奇异值达到 1.05 也能收敛。

**避免显式添加单位矩阵**

计算矩阵二次型 $\mathbf{Z}_t \gets a_t\mathbf{I} + b_t \mathbf{R}_{t-1} + c_t \mathbf{R}_{t-1}^2$ 时，显式添加 $a_t\mathbf{I}$ 可能降低稳定性。更准确的方式是：

$$
\begin{aligned}
\mathbf{Z}_t &\gets b_t \mathbf{R}_{t-1} + c_t \mathbf{R}_{t-1}^2 \\
\mathbf{Q}_t &\gets \mathbf{Q}_{t-1} \mathbf{Z}_t + a_t\mathbf{Q}_{t-1}
\end{aligned}
$$

这样所有涉及 $a_t$ 的算术都在 float32 中进行，精度更高。

### 3.4. Stabilized Gram Newton-Schulz

综合以上分析，作者提出了完整的 Stabilized Gram Newton-Schulz 算法：

**算法 2（Stabilized Gram Newton-Schulz）**

输入：$\mathbf{X} \in \mathbb{R}^{n \times m}$（$n \leq m$），系数 $\{(a_t, b_t, c_t)\}_{t=1}^5$

1. $\mathbf{X} \gets \mathbf{X} / (\Vert\mathbf{X}\Vert_{\mathsf{F}} + \epsilon)$ 
2. $\mathbf{X} \gets \text{float16}(\mathbf{X})$  （转换为半精度）
3. **第一次迭代**：
   - $\mathbf{R} \gets \mathbf{X}\mathbf{X}^\top$
   - $\mathbf{Q} \gets \mathbf{I}$
   - 对于 $t = 1, 2$：
     - $\mathbf{Z} \gets b_t \mathbf{R} + c_t \mathbf{R}^2$
     - $\mathbf{Q} \gets \mathbf{Q}\mathbf{Z} + a_t\mathbf{Q}$
     - $\mathbf{RZ} \gets \mathbf{R}\mathbf{Z} + a_t\mathbf{R}$
     - $\mathbf{R} \gets \mathbf{Z}(\mathbf{RZ}) + a_t(\mathbf{RZ})$
   - $\mathbf{X} \gets \mathbf{Q}\mathbf{X}$  （重启）
4. **第二次迭代**：
   - $\mathbf{R} \gets \mathbf{X}\mathbf{X}^\top$
   - $\mathbf{Q} \gets \mathbf{I}$
   - 对于 $t = 3, 4, 5$：
     - $\mathbf{Z} \gets b_t \mathbf{R} + c_t \mathbf{R}^2$
     - $\mathbf{Q} \gets \mathbf{Q}\mathbf{Z} + a_t\mathbf{Q}$
     - $\mathbf{RZ} \gets \mathbf{R}\mathbf{Z} + a_t\mathbf{R}$
     - $\mathbf{R} \gets \mathbf{Z}(\mathbf{RZ}) + a_t(\mathbf{RZ})$
5. 返回 $\mathbf{Q}\mathbf{X}$

下面来计算 Stabilized Gram Newton-Schulz 的计算复杂度（带一次重启）。

一次重启需要额外两次矩阵乘法：
- $\mathbf{X} \gets \mathbf{Q}_2 \mathbf{X}$：$2mn^2$ FLOPs
- $\mathbf{R}_2 \gets \mathbf{X}\mathbf{X}^\top$：$mn^2$ FLOPs

同时可以省略三次乘法：
- $\mathbf{R}_2 \gets \mathbf{Z}_2 \mathbf{R}_1 \mathbf{Z}_2$：节省 $2n^3$ FLOPs
- $\mathbf{Q}_3 \gets \mathbf{Q}_2 \mathbf{Z}_3$：节省 $n^3$ FLOPs

因此，带一次重启的 Stabilized Gram Newton-Schulz 的计算复杂度为：

$$
\begin{equation}
(4T + 6\alpha - 6)n^3 \text{ FLOPs}
\end{equation}
$$

对于 $T=5, \alpha=4$：
- 相比使用对称 GEMM 的标准 Newton-Schulz，减少 **42%** FLOPs
- 相比典型实现（无对称 GEMM），减少 **58%** FLOPs

此外，如果使用 $T-1$ 次重启，Gram Newton-Schulz 就完全退化为标准 Newton-Schulz。因此，添加重启可以视为在运行时间和稳定性之间权衡。

## 四、对称 GEMM 核优化

为了充分利用 Gram Newton-Schulz 带来的对称矩阵乘法机会，作者实现了专门的对称 GEMM 核。

1. 三角调度器 (Triangular Scheduler)：作者仅将下三角部分的工作块分配给 warp group，上三角部分的值通过对称性获得（$\mathbf{C}_{ij} = \mathbf{C}_{ji}$）。
2. 融合二次型核：对称 GEMM 核可以融合矩阵二次型的计算，通过在寄存器级别添加 $\gamma$ 到所有对角线元素，完全避免了加载单位矩阵的 I/O 操作。
   $$
   \mathbf{C} \gets \alpha \mathbf{A}\mathbf{B} + \beta \mathbf{C} + \gamma \mathbf{I}
   $$

## 五、实验结果

### 5.1. 数值稳定性验证

在 Llama-430M 模型上的实验表明：
- **Naive Gram Newton-Schulz**：出现损失尖峰，最终输出充满 Inf
- **Stabilized Gram Newton-Schulz**：训练稳定，验证困惑度差异在 0.01 以内

### 5.2. 加速效果

在 Kimi K2 上：
- Newton-Schulz 正交化步骤运行时间减少 **40-50%**
- 端到端训练时间减少 **2-17%**（取决于训练配置）

加速比随矩阵纵横比 $\alpha = m/n$ 增加而提高：
- $\alpha = 1$（方阵）：无加速（退化为标准 Newton-Schulz）
- $\alpha = 4$（典型 MLP）：**42%** FLOP 减少
- $\alpha = 8$（细粒度 MoE）：**55%** FLOP 减少

### 5.3. 对称 GEMM 核性能

在 Hopper (H100) 和 Blackwell (B200) GPU 架构上 benchmark：
- 相比 cuBLAS，对称 GEMM 核实现 **1.5-2x** 加速
- 在 Gram Newton-Schulz 中贡献了额外的 **10-15%** 加速
- 对于 $256 \times 256$ 以上矩阵，核效率超过 80%

## Appendix

### Apd.1. Proof of Proposition 1

**Proposition 1（奇异向量不变性）**：设 $\mathbf{X}_0 = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$ 是 $\mathbf{X}_0$ 的 SVD 分解。若 $\mathbf{X}_t$ 按照方程 \eqref{eq:newton-schulz} 迭代，则对任意 $t \geq 0$，存在奇次多项式 $p_t: \mathbb{R} \to \mathbb{R}$ 使得：

$$
\mathbf{X}_t = \mathbf{U} p_t(\mathbf{\Sigma}) \mathbf{V}^\top
$$

其中 $p_t(\mathbf{\Sigma}) = \text{diag}(p_t(\sigma_1), \ldots, p_t(\sigma_r))$，$r$ 为 $\mathbf{X}_0$ 的秩。

**Proof**：我们使用数学归纳法。

当 $t=0$ 时，结论显然成立。

假设当 $t=k$ 时，$\mathbf{X}_k = \mathbf{U} p_k(\mathbf{\Sigma}) \mathbf{V}^\top$，其中 $p_k$ 是某个奇次多项式。

则当 $t=k+1$ 时：

$$
\begin{equation}
\begin{aligned}
\mathbf{X}_{k+1} &= a_{k+1} \mathbf{X}_k + b_{k+1} \mathbf{X}_k \mathbf{X}_k^\top \mathbf{X}_k + c_{k+1} \left(\mathbf{X}_k \mathbf{X}_k^\top\right)^2 \mathbf{X}_k \\
&= a_{k+1} \mathbf{U} p_k(\mathbf{\Sigma}) \mathbf{V}^\top + b_{k+1} \mathbf{U} p_k(\mathbf{\Sigma})^3 \mathbf{V}^\top + c_{k+1} \mathbf{U} p_k(\mathbf{\Sigma})^5 \mathbf{V}^\top \\
&= \mathbf{U} \left[ a_{k+1} p_k(\mathbf{\Sigma}) + b_{k+1} p_k(\mathbf{\Sigma})^3 + c_{k+1} p_k(\mathbf{\Sigma})^5 \right] \mathbf{V}^\top \\
&= \mathbf{U} p_{k+1}(\mathbf{\Sigma}) \mathbf{V}^\top
\end{aligned}
\end{equation}
$$

其中 $p_{k+1}(x) = a_{k+1} p_k(x) + b_{k+1} p_k(x)^3 + c_{k+1} p_k(x)^5$。

因此结论对 $t=k+1$ 也成立。由数学归纳法，命题得证。$\blacksquare$

### Apd.2. Proof of Theorem 1

**Theorem 1（迭代求逆平方根）**：设任取 $t \in \{1, \ldots, T\}$ 都有

$$
\begin{equation}
    p_t(x) = x h_t(x^2)
\end{equation}
$$

则有

$$
\begin{equation}
    (p_T \circ \cdots \circ p_1)(x) = q_T x
\end{equation}
$$

其中 $q_T$ 由以下迭代定义：

$$
\begin{equation}
\begin{cases}
z_t = h_t(r_{t-1}) \\
r_t = r_{t-1} z_t^2 \\
q_t = q_{t-1} z_t
\end{cases}
\end{equation}
$$

迭代的起始条件为 $r_0 = x^2$，$q_0 = 1$。

**Proof**：任取对 $t \in \{1, \ldots, T\}$，定义 $x_0 = x$ 且 $x_t = p_t(x_{t-1})$。我们将通过归纳法证明 $r_t = x_t^2$ 且 $q_t = x_t/x_0$ 对所有 $t$ 成立。

当 $t=0$ 时，由定义 $r_0 = x^2$，$q_0 = 1$，结论成立。

假设结论对 $t-1$ 成立，即 $r_{t-1} = x_{t-1}^2$ 且 $q_{t-1} = x_{t-1}/x_0$。

由假设 $p_t(x) = x h_t(x^2)$，我们有：

$$
x_t = p_t(x_{t-1}) = x_{t-1} h_t(x_{t-1}^2)
$$

注意到 $h_t(x_{t-1}^2) = h_t(r_{t-1}) = z_t$，因此 $x_t = x_{t-1} z_t$。

两边平方得：

$$
x_t^2 = x_{t-1}^2 z_t^2 = r_{t-1} z_t^2 = r_t
$$

两边除以 $x_0$ 得：

$$
\frac{x_t}{x_0} = \frac{x_{t-1}}{x_0} z_t = q_{t-1} z_t = q_t
$$

因此结论对 $t$ 也成立。由数学归纳法，定理得证。$\blacksquare$



