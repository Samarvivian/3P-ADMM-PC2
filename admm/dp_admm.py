import numpy as np
from numpy.linalg import inv

def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def dp_admm(A, y, K=3, rho=1.0, lam=1.0, max_iter=100, tol=1e-4, sigma=0.01):
    """
    差分隐私ADMM
    sigma: 高斯噪声标准差，控制隐私保护强度
    """
    M, N = A.shape
    Nk = N // K

    A_blocks = [A[:, k*Nk:(k+1)*Nk] for k in range(K)]

    x = np.zeros(N)
    z = np.zeros(N)
    v = np.zeros(N)

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

        for k in range(K):
            Ak = A_blocks[k]
            Bk = B_blocks[k]
            zk = z[k*Nk:(k+1)*Nk]
            vk = v[k*Nk:(k+1)*Nk]
            xk = Bk @ (Ak.T @ y / K + rho * (zk - vk))
            # 加入高斯噪声实现差分隐私
            xk += np.random.normal(0, sigma, xk.shape)
            x_new[k*Nk:(k+1)*Nk] = xk

        x = x_new
        z = soft_threshold(v + x, lam / rho)
        v = v + x - z

        mse = np.mean((A @ x - y) ** 2)
        mse_list.append(mse)

        if t % 10 == 0:
            print(f"迭代 {t}: MSE = {mse:.6f}")

        if mse < tol:
            print(f"第 {t} 轮收敛")
            break

    return x, mse_list
