import numpy as np
import requests
import time
import argparse
import matplotlib.pyplot as plt
import os

# ---------- CONFIG ----------
EDGE_URLS = [
    "http://192.168.130.223:5000",
]
K = len(EDGE_URLS)
RHO = 1.0
LAMBDA = 1.0
MAX_ITER = 100
SEED = 42
MSE_TOL = 1e-3

# ---------- UTIL ----------
def post(url, path, js, timeout=10):
    try:
        r = requests.post(url.rstrip('/') + path, json=js, timeout=timeout)
        return r.json()
    except Exception as e:
        raise RuntimeError(f"POST {url}{path} failed: {e}")

def soft_threshold(u, kappa):
    return np.sign(u) * np.maximum(np.abs(u) - kappa, 0.0)

# ---------- PROBLEM SETUP ----------
def build_problem(M=200, N=600, sparsity=0.1, noise_sigma=0.1, seed=SEED):
    np.random.seed(seed)
    k = int(np.floor(sparsity * N))
    A = np.random.randn(M, N) / np.sqrt(M)
    x_true = np.zeros(N)
    support = np.random.choice(N, k, replace=False)
    x_true[support] = np.random.randn(k)
    y = A @ x_true + np.random.normal(loc=0, scale=noise_sigma, size=A.shape[0])
    splits = np.array_split(np.arange(N), K)
    A_parts = [A[:, idx] for idx in splits]
    return A, A_parts, y, x_true, splits

# ---------- MAIN MASTER LOGIC ----------
def master_run():
    A, A_parts, y, x_true, splits = build_problem(M=200, N=600, sparsity=0.1, noise_sigma=0.1)
    Ns = [Ap.shape[1] for Ap in A_parts]

    # --- Initialization Phase ---
    B_list = []
    print("[MASTER] Initialization: send A_k^T A_k and rho to edges, receive B_k")
    for k, url in enumerate(EDGE_URLS):
        Ak = A_parts[k]
        AtA = Ak.T @ Ak
        payload = {'AtA': AtA.tolist(), 'rho': RHO}
        res = post(url, '/init_atat', payload)
        if not res.get('ok'):
            raise RuntimeError(f"Edge {k} init_atat failed: {res}")
        Bk = np.array(res['Bk'], dtype=float)
        B_list.append(Bk)

    print("[MASTER] Computing alpha_k and Bbar_k; sending to edges")
    for k, url in enumerate(EDGE_URLS):
        Ak = A_parts[k]
        At_y = Ak.T @ y
        Bk = B_list[k]
        alpha_k = Bk.dot(At_y)
        Bbar_k = RHO * Bk
        payload = {'alpha': alpha_k.tolist(), 'Bbar': Bbar_k.tolist()}
        res = post(url, '/init_params', payload)
        if not res.get('ok'):
            raise RuntimeError(f"Edge {k} init_params failed: {res}")

    # initialize x,z,v
    x_parts = [np.zeros(Ns[k]) for k in range(K)]
    z_parts = [np.zeros(Ns[k]) for k in range(K)]
    v_parts = [np.zeros(Ns[k]) for k in range(K)]

    mse_history = []

    # Iterations
    for t in range(1, MAX_ITER + 1):
        x_new_parts = []
        for k, url in enumerate(EDGE_URLS):
            payload = {'z_k': z_parts[k].tolist(), 'v_k': v_parts[k].tolist()}
            res = post(url, '/compute_x', payload, timeout=30)
            if not res.get('ok'):
                raise RuntimeError(f"Edge {k} compute_x failed: {res}")
            xk = np.array(res['x_k'], dtype=float)
            x_new_parts.append(xk)

        x_full = np.concatenate(x_new_parts)
        v_full = np.concatenate(v_parts)
        z_full = np.concatenate(z_parts)
        z_full = soft_threshold(v_full + x_full, LAMBDA / RHO)
        v_full = v_full + x_full - z_full

        offset = 0
        for k in range(K):
            n = Ns[k]
            z_parts[k] = z_full[offset:offset+n].copy()
            v_parts[k] = v_full[offset:offset+n].copy()
            offset += n
            x_parts[k] = x_new_parts[k]

        mse = np.mean((x_full - x_true) ** 2)
        mse_history.append(mse)
        print(f"[MASTER] iter {t:4d}  MSE = {mse:.3e}")

        # if mse < MSE_TOL:
        #     print(f"[MASTER] Early stop: MSE < {MSE_TOL:.1e} at iter {t}")
        #     break

    # ---------- 保存结果到 .npy ----------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    np.save(f"Dis-ADMM{timestamp}.npy", np.array(mse_history))
    np.save(f"x_est_final_{timestamp}.npy", x_full)
    np.save(f"x_true_{timestamp}.npy", x_true)
    print(f"[MASTER] Saved: mse_history_{timestamp}.npy, x_est_final_{timestamp}.npy, x_true_{timestamp}.npy")

    # ---------- 绘制并保存 MSE 曲线 ----------
    plt.figure(figsize=(8,4))
    plt.semilogy(range(1, len(mse_history)+1), mse_history,
                 'o-', linewidth=2, label='Dis-ADMM')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('MSE (mean squared error)', fontsize=12)
    plt.title('Dis-ADMM — MSE vs Iteration')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    fig_name = f"Dis_ADMM.png"
    plt.savefig(fig_name, dpi=300)
    print(f"[MASTER] MSE curve saved as {fig_name}")

    plt.show()

    print(f"Final MSE after {t} iterations: {mse_history[-1]:.3e}")
    return mse_history

if __name__ == '__main__':
    master_run()
