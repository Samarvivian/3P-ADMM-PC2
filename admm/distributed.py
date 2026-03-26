import numpy as np
from numpy.linalg import inv

def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def dis_admm(A, y, K=3, rho=1.0, lam=1.0, max_iter=100, tol=1e-4):
    """
    分布式ADMM
    A: M x N 观测矩阵
    y: M 观测向量
    K: 边缘节点数量
    """
    M, N = A.shape
    Nk = N // K  # 每个节点的子问题维度

    # 按列分块 A = [A1, A2, ..., AK]
    A_blocks = [A[:, k*Nk:(k+1)*Nk] for k in range(K)]

    # 初始化
    x = np.zeros(N)
    z = np.zeros(N)
    v = np.zeros(N)

    # 预计算每个节点的 Bk = (Ak^T Ak + rho I)^{-1}
    print("预计算各节点矩阵求逆...")
    B_blocks = []
    for k in range(K):
        Ak = A_blocks[k]
        Bk = inv(Ak.T @ Ak + rho * np.eye(Nk))
        B_blocks.append(Bk)
    print("预计算完成")

    mse_list = []

    for t in range(max_iter):
        x_new = np.zeros(N)

        # 各节点并行更新 xk
        for k in range(K):
            Ak = A_blocks[k]
            Bk = B_blocks[k]
            zk = z[k*Nk:(k+1)*Nk]
            vk = v[k*Nk:(k+1)*Nk]
            # xk = Bk @ (Ak^T y/K + rho(zk - vk))
            xk = Bk @ (Ak.T @ y / K + rho * (zk - vk))
            x_new[k*Nk:(k+1)*Nk] = xk

        x = x_new

        # z更新（主节点）
        z = soft_threshold(v + x, lam / rho)

        # v更新（主节点）
        v = v + x - z

        # 计算MSE
        mse = np.mean((A @ x - y) ** 2)
        mse_list.append(mse)

        if t % 10 == 0:
            print(f"迭代 {t}: MSE = {mse:.6f}")

        if mse < tol:
            print(f"第 {t} 轮收敛")
            break

    return x, mse_list
