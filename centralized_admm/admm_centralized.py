#!/usr/bin/env python3
"""
集中式 ADMM（用于 LASSO）示例

目标问题：
    minimize (1/2)||X w - y||_2^2 + lambda * ||z||_1  subject to w = z

ADMM 变量：w, z, u

输出：每次迭代的估计向量和 MSE（相对于真实参数）随迭代变化
"""
import numpy as np
import matplotlib.pyplot as plt
import time


def soft_thresholding(x, kappa):
    return np.sign(x) * np.maximum(np.abs(x) - kappa, 0.0)


def admm_lasso(X, y, lam=1.0, rho=1.0, max_iter=100, abstol=1e-4, reltol=1e-3, w_true=None):
    n, p = X.shape
    # variables
    w = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)

    # Precompute factorization for w-update: (X^T X + rho I) w = X^T y + rho (z - u)
    XtX = X.T.dot(X)
    Xty = X.T.dot(y)
    A = XtX + rho * np.eye(p)
    # use cholesky if possible
    try:
        L = np.linalg.cholesky(A)
        use_cholesky = True
    except np.linalg.LinAlgError:
        use_cholesky = False

    mse_list = []

    for k in range(1, max_iter + 1):
        # w-update (quadratic minimization)
        b = Xty + rho * (z - u)
        if use_cholesky:
            # solve A w = b using cholesky
            y_ = np.linalg.solve(L, b)
            w = np.linalg.solve(L.T, y_)
        else:
            w = np.linalg.solve(A, b)

        # z-update (soft-thresholding)
        z_old = z.copy()
        z = soft_thresholding(w + u, lam / rho)

        # u-update (dual variable)
        u = u + w - z

        # diagnostics, compute residuals
        r_norm = np.linalg.norm(w - z)
        s_norm = np.linalg.norm(-rho * (z - z_old))

        eps_pri = np.sqrt(p) * abstol + reltol * max(np.linalg.norm(w), np.linalg.norm(z))
        eps_dual = np.sqrt(p) * abstol + reltol * np.linalg.norm(rho * u)

        if w_true is not None:
            mse = float(np.mean((z - w_true) ** 2))
            mse_list.append(mse)

        # stopping criterion (optional)
        if r_norm <= eps_pri and s_norm <= eps_dual:
            # append remaining iterations with same mse if any
            break

    return {
        'w': w,
        'z': z,
        'u': u,
        'mse_list': mse_list,
        'iters': k,
    }


def simulate_and_run(seed=0):
    np.random.seed(seed)
    n = 200  # number of samples
    p = 50   # number of features
    k = 8    # sparsity (non-zero entries in true w)

    # generate true sparse parameter
    w_true = np.zeros(p)
    nonzero_idx = np.random.choice(p, k, replace=False)
    w_true[nonzero_idx] = np.random.randn(k) * 5.0

    # design matrix and observations
    X = np.random.randn(n, p)
    sigma = 1.0
    noise = sigma * np.random.randn(n)
    y = X.dot(w_true) + noise

    lam = 1.0  # L1 regularization weight
    rho = 1.0
    max_iter = 200

    t0 = time.time()
    out = admm_lasso(X, y, lam=lam, rho=rho, max_iter=max_iter, w_true=w_true)
    t1 = time.time()

    mse_list = out['mse_list']
    iters = out['iters']

    print(f"Ran ADMM for {iters} iterations, time used: {t1 - t0:.3f}s")
    if len(mse_list) > 0:
        print(f"Final MSE (w_true vs z): {mse_list[-1]:.6e}")
    else:
        print("No MSE recorded (w_true not provided or 0 iters)")

    # 绘制 MSE 迭代曲线
    if len(mse_list) > 0:
        plt.figure(figsize=(6, 4))
        plt.semilogy(range(1, len(mse_list) + 1), mse_list, marker='o')
        plt.xlabel('ADMM iteration')
        plt.ylabel('MSE (z vs w_true)')
        plt.title('MSE per ADMM iteration (LASSO via centralized ADMM)')
        plt.grid(True, which='both', ls='--', lw=0.5)
        out_png = 'mse_iterations.png'
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved MSE plot to {out_png}")

    return out, w_true


if __name__ == '__main__':
    out, w_true = simulate_and_run(seed=42)
