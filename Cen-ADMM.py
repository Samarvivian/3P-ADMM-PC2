# centralized_admm_gpu.py
"""
Centralized ADMM for LASSO (GPU-driven via CuPy).
Now saves result files to project root (PyCharm auto-sync to local).
"""

import os
import math
import time
import numpy as np
import cupy as cp
import cupyx.scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def gpu_mem_info():
    free, total = cp.cuda.runtime.memGetInfo()
    return int(free), int(total)


def soft_threshold(u, kappa):
    return cp.sign(u) * cp.maximum(cp.abs(u) - kappa, 0.0)


def solve_direct_cholesky(A, rho, b):
    N = A.shape[1]
    AtA = A.T @ A
    P = AtA + rho * cp.eye(N, dtype=A.dtype)
    L = cp.linalg.cholesky(P)
    y = cp.linalg.solve(L, b)
    x = cp.linalg.solve(L.T, y)
    del AtA, P, L, y
    cp._default_memory_pool.free_all_blocks()
    return x


def solve_cg_matvec(A, rho, b, x0=None, tol=1e-6, maxiter=1000):
    M, N = A.shape

    def mv(x_vec):
        x_cp = cp.asarray(x_vec)
        Ax = A @ x_cp
        AtAx = A.T @ Ax
        return AtAx + rho * x_cp

    Aop = spla.LinearOperator((N, N), matvec=mv, dtype=A.dtype)
    if x0 is None:
        x0 = cp.zeros(N, dtype=A.dtype)
    x, info = spla.cg(Aop, b, x0=x0, tol=tol, maxiter=maxiter)
    if info != 0:
        print(f"[CG] exited with info={info}")
    return x


def centralized_admm_lasso_gpu(A, y, x_true, rho=1.0, lamb=1.0, max_iter=100,
                              solver_pref='auto', dtype=cp.float32):
    A = cp.asarray(A, dtype=dtype)
    y = cp.asarray(y, dtype=dtype)
    x_true = cp.asarray(x_true, dtype=dtype)
    M, N = A.shape

    print(f"[INFO] A shape: {M} x {N}, dtype={A.dtype}")
    x = cp.zeros(N, dtype=dtype)
    z = cp.zeros(N, dtype=dtype)
    v = cp.zeros(N, dtype=dtype)
    history_x = []
    mse_list = []

    Aty = A.T @ y

    chosen_solver = solver_pref
    if solver_pref == 'auto':
        free, total = gpu_mem_info()
        bytes_needed = N * N * cp.dtype(dtype).itemsize * 1.1
        print(f"[MEM] GPU free {free/1e9:.2f} GB, total {total/1e9:.2f} GB")
        print(f"[MEM] approx bytes for AtA: {bytes_needed/1e9:.2f} GB")
        chosen_solver = 'direct' if bytes_needed < free * 0.6 else 'cg'
    print(f"[INFO] solver selected: {chosen_solver}")

    x0 = None
    for t in range(1, max_iter + 1):
        b = Aty + rho * (z - v)
        t0 = time.time()
        try:
            if chosen_solver == 'direct':
                x = solve_direct_cholesky(A, rho, b)
            else:
                x = solve_cg_matvec(A, rho, b, x0=x0, tol=1e-6, maxiter=5000)
            cg_time = time.time() - t0
        except cp.cuda.memory.OutOfMemoryError as e:
            print(f"[OOM] Direct solver OOM, switching to CG. {e}")
            cp._default_memory_pool.free_all_blocks()
            chosen_solver = 'cg'
            x = solve_cg_matvec(A, rho, b, x0=x0, tol=1e-6, maxiter=5000)
            cg_time = time.time() - t0

        x0 = x
        z = soft_threshold(v + x, lamb / rho)
        v = v + x - z

        history_x.append(cp.asnumpy(x))
        mse = float(cp.asnumpy(cp.mean((x - x_true) ** 2)))
        mse_list.append(mse)
        free, total = gpu_mem_info()
        print(f"[ITER {t:03d}] MSE={mse:.6e} | solver={chosen_solver} | time={cg_time:.3f}s | free={free/1e9:.2f}GB")

    return history_x, mse_list


def experiment_and_plot_gpu(seed=42, M=200, N=600, sparsity=0.1,
                            rho=1.0, lamb=0.1, noise_sigma=0.01, max_iter=100,
                            solver_pref='auto', dtype=cp.float32):
    np.random.seed(seed)
    k = int(math.floor(sparsity * N))
    A_cpu = np.random.randn(M, N).astype(np.float32) / math.sqrt(M)
    x_true_cpu = np.zeros(N, dtype=np.float32)
    support = np.random.choice(N, k, replace=False)
    x_true_cpu[support] = np.random.randn(k).astype(np.float32)
    y_cpu = A_cpu @ x_true_cpu + noise_sigma * np.random.randn(M).astype(np.float32)

    A = cp.asarray(A_cpu, dtype=dtype)
    y = cp.asarray(y_cpu, dtype=dtype)
    x_true = cp.asarray(x_true_cpu, dtype=dtype)

    hist_x, mse = centralized_admm_lasso_gpu(
        A, y, x_true, rho=rho, lamb=lamb, max_iter=max_iter,
        solver_pref=solver_pref, dtype=dtype
    )


    project_dir = os.getcwd()
    ts = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(project_dir, f"Cen-ADMM-GPU_{ts}.png")
    npy_path = os.path.join(project_dir, f"Cen-ADMM-MSE-GPU_{ts}.npy")

    # 绘图与保存
    plt.figure(figsize=(8,4))
    plt.semilogy(range(1, len(mse)+1), mse, label='Centralized ADMM (GPU)')
    plt.xlabel('Iteration')
    plt.ylabel('MSE (mean squared error)')
    plt.title('Centralized ADMM (GPU) — MSE vs Iteration')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    np.save(npy_path, np.array(mse))

    print(f"[Saved] Plot → {save_path}")
    print(f"[Saved] MSE  → {npy_path}")
    print(f"Final MSE after {max_iter} iterations: {mse[-1]:.3e}")
    plt.show()
    return mse


if __name__ == "__main__":
    _mse = experiment_and_plot_gpu(
        seed=42,
        M=200,
        N=1600,
        sparsity=0.1,
        rho=1.0,
        lamb=0.1,
        noise_sigma=0.01,
        max_iter=100,
        solver_pref='auto',
        dtype=cp.float32
    )
