import numpy as np
from numpy.linalg import inv

def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def cen_admm(A, y, rho=1.0, lam=1.0, max_iter=100, tol=1e-4):
    M, N = A.shape
    x = np.zeros(N)
    z = np.zeros(N)
    v = np.zeros(N)

    # 预计算 (A^T A + rho I)^{-1}
    print("预计算矩阵求逆...")
    ATA = A.T @ A
    B = inv(ATA + rho * np.eye(N))
    ATy = A.T @ y
    print("预计算完成")

    mse_list = []

    for t in range(max_iter):
        # x更新
        x = B @ (ATy + rho * (z - v))
        # z更新
        z = soft_threshold(v + x, lam / rho)
        # v更新
        v = v + x - z

        # 计算MSE
        mse = np.mean((A @ x - y) ** 2)
        mse_list.append(mse)

        if t % 10 == 0:
            print(f"迭代 {t}: MSE = {mse:.6f}")

        # 收敛判断
        if mse < tol:
            print(f"第 {t} 轮收敛")
            break

    return x, mse_list
