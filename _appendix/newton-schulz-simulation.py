"""
这个脚本实现了基于Newton-Schulz迭代方法的矩阵求逆优化算法模拟。
主要功能包括：
1. 生成随机矩阵并计算其奇异值分布
2. 定义基于Newton-Schulz迭代的目标函数
3. 使用动量SGD优化算法参数
4. 输出优化结果和性能指标
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def generate_singular_value_data(n=1024, m=1024, num_samples=1000):
    """
    生成随机矩阵并计算其归一化奇异值数据
    
    参数:
        n: 矩阵行数
        m: 矩阵列数  
        num_samples: 生成的矩阵数量
    
    返回:
        data: 包含所有归一化奇异值的张量
    """
    data_list = []
    
    # 使用进度条显示生成过程
    for _ in tqdm(range(num_samples), ncols=0, desc='生成奇异值数据'):
        # 生成随机正态分布矩阵
        M = torch.randn(n, m)
        
        # 计算奇异值分解 (SVD)
        _, S, _ = torch.svd(M, some=True)
        
        # 归一化奇异值: S / ||S||_2
        normalized_S = S / torch.norm(S, p=2)
        
        data_list.append(normalized_S)
    
    # 将所有奇异值数据合并成一个张量
    data = torch.cat(data_list)
    return data


def newton_schulz_iteration(x, k, x1, x2, num_iterations=5):
    """
    Newton-Schulz迭代函数
    
    实现迭代公式: x = x + k * x * (x^2 - x1^2) * (x^2 - x2^2)
    
    参数:
        x: 输入张量
        k: 学习率参数
        x1, x2: 迭代参数
        num_iterations: 迭代次数
    
    返回:
        迭代后的张量
    """
    for _ in range(num_iterations):
        # Newton-Schulz迭代公式
        x = x + k * x * (x**2 - x1**2) * (x**2 - x2**2)
    
    return x


def objective_function(params, data, num_iterations=5):
    """
    目标函数: 衡量Newton-Schulz迭代后的结果与目标值(1)的差距
    
    参数:
        params: 包含k, x1, x2三个参数的张量
        data: 输入数据（奇异值）
        num_iterations: Newton-Schulz迭代次数
    
    返回:
        loss: 均方误差损失
    """
    # 解包参数
    k, x1, x2 = params
    
    # 应用Newton-Schulz迭代
    x_transformed = newton_schulz_iteration(data, k, x1, x2, num_iterations)
    
    # 计算与目标值1的均方误差
    loss = torch.mean((x_transformed - 1.0) ** 2)
    
    return loss


def momentum_sgd_optimization(initial_params, data, learning_rate=0.01, momentum=0.9, num_iterations=100000):
    """
    动量随机梯度下降优化
    
    参数:
        initial_params: 初始参数 [k, x1, x2]
        data: 训练数据
        learning_rate: 学习率
        momentum: 动量系数
        num_iterations: 优化迭代次数
    
    返回:
        optimized_params: 优化后的参数
        loss_history: 损失历史记录
    """
    # 将参数转换为可训练的张量
    params = torch.tensor(initial_params, dtype=torch.float32, requires_grad=True)
    
    # 初始化动量项
    velocity = torch.zeros_like(params)
    
    # 记录损失历史
    loss_history = []
    
    # 优化过程
    for i in tqdm(range(num_iterations), ncols=0, desc='动量SGD优化'):
        # 清零梯度
        if params.grad is not None:
            params.grad.zero_()
        
        # 计算损失
        loss = objective_function(params, data)
        
        # 反向传播计算梯度
        loss.backward()
        
        # 动量更新
        velocity = momentum * velocity + params.grad
        
        # 参数更新（原地操作）
        with torch.no_grad():
            params -= learning_rate * velocity
        
        # 记录损失
        if i % 1000 == 0:  # 每1000次迭代记录一次
            loss_history.append(loss.item())
    
    return params.detach(), loss_history


def simulate(n=1024, m=1024, T=5):
    """主函数：执行完整的Newton-Schulz模拟流程"""
    
    print("=== Newton-Schulz迭代方法模拟 ===")
    print(f"矩阵维度: {n} x {m}")
    print(f"迭代次数: {T}")
    print()
    
    # 步骤1: 生成奇异值数据
    print("步骤1: 生成奇异值数据...")
    data = generate_singular_value_data(n, m, num_samples=1000)
    print(f"生成数据形状: {data.shape}")
    print()
    
    # 步骤2: 设置初始参数
    initial_params = [1.0, 0.9, 1.1]  # [k, x1, x2]
    print("步骤2: 初始化参数...")
    print(f"初始参数: k={initial_params[0]}, x1={initial_params[1]}, x2={initial_params[2]}")
    print()
    
    # 步骤3: 优化参数
    print("步骤3: 开始优化过程...")
    optimized_params, loss_history = momentum_sgd_optimization(
        initial_params, data, learning_rate=0.01, momentum=0.9, num_iterations=100000
    )
    
    # 解包优化后的参数
    k, x1, x2 = optimized_params
    
    # 步骤4: 计算最终损失和性能指标
    final_loss = objective_function(optimized_params, data, T).item()
    
    # 计算相关参数
    a = 1 + k * x1**2 * x2**2
    b = -k * (x1**2 + x2**2)
    c = k
    
    # 步骤5: 输出结果
    print("\n=== 优化结果 ===")
    print(f"优化后的参数:")
    print(f"  k = {k:.6f}")
    print(f"  x1 = {x1:.6f}")
    print(f"  x2 = {x2:.6f}")
    print()
    print(f"性能指标:")
    print(f"  a = {a:.6f}")
    print(f"  b = {b:.6f}")
    print(f"  c = {c:.6f}")
    print(f"  最终损失 = {final_loss:.8f}")
    print()
    
    # 格式化输出（用于表格）
    print("=== 表格格式输出 ===")
    print(f"{n} & {m} & {T} & {k:.3f} & {x1:.3f} & {x2:.3f} & {a:.3f} & {b:.3f} & {c:.3f} & {final_loss:.5f}")
    
    return optimized_params, final_loss, loss_history


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 执行主程序
    for n in [1024, 2048, 4096, 8192]:
        for m in [1024, 2048, 4096, 8192]:
            for T in [3, 5, 10]:
                optimized_params, final_loss, loss_history = simulate(n, m, T)
    
    # 可选：绘制损失曲线
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.title('Newton-Schulz优化损失曲线')
        plt.xlabel('迭代次数 (x1000)')
        plt.ylabel('损失值')
        plt.grid(True)
        plt.show()
        
    except ImportError:
        print("\n注意: 未安装matplotlib，无法绘制损失曲线")
        print("如需绘图，请运行: pip install matplotlib")