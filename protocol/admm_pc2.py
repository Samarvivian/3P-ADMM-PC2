import numpy as np
from numpy.linalg import inv
import sys
sys.path.append('/mnt/3p-admm-pc2')
from crypto.paillier import generate_keypair, encrypt, decrypt, homo_add, homo_mul_const

def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def quantize2(v, delta, zmin, zmax):
    return np.floor(delta * (v - zmin) / (zmax - zmin)).astype(object)

def quantize1(v, delta, zmin, zmax):
    return np.floor(delta**2 * (v - zmin) / (zmax - zmin)**2).astype(object)

def admm_pc2(A, y, K=3, rho=1.0, lam=1.0, max_iter=100, delta=10**10, bits=1024):
    M, N = A.shape
    Nk = N // K
    A_blocks = [A[:, k*Nk:(k+1)*Nk] for k in range(K)]

    # 初始化阶段
    print("【初始化】生成密钥...")
    pub, priv = generate_keypair(bits=bits)

    print("【初始化】各节点预计算矩阵求逆...")
    B_blocks = []
    alpha_ks = []
    for k in range(K):
        Ak = A_blocks[k]
        Bk = inv(Ak.T @ Ak + rho * np.eye(Nk))
        B_blocks.append(Bk)
        alpha_k = Bk @ (Ak.T @ y)
        alpha_ks.append(alpha_k)

    # 数据安全共享阶段：加密 alpha_k
    print("【数据安全共享】加密 alpha_k...")
    alpha_hats = []
    alpha_ranges = []
    for k in range(K):
        Bk_diag = np.diag(B_blocks[k])
        zk = np.zeros(Nk)
        vk = np.zeros(Nk)
        all_vals = np.concatenate([alpha_ks[k], zk, -vk, Bk_diag])
        zmin = all_vals.min()
        zmax = all_vals.max()
        if zmax == zmin:
            zmax = zmin + 1e-10

        q_alpha = quantize1(alpha_ks[k], delta, zmin, zmax)
        alpha_hat = [encrypt(int(qi), pub) for qi in q_alpha]
        alpha_hats.append(alpha_hat)
        alpha_ranges.append((zmin, zmax))

    # 并行隐私计算阶段
    print("【迭代计算】开始...")
    x = np.zeros(N)
    z = np.zeros(N)
    v = np.zeros(N)
    mse_list = []

    for t in range(max_iter):
        x_new = np.zeros(N)

        for k in range(K):
            zk = z[k*Nk:(k+1)*Nk]
            vk = v[k*Nk:(k+1)*Nk]
            Bk = B_blocks[k]
            Bk_diag = np.diag(Bk)

            # 统一量化范围
            all_vals = np.concatenate([alpha_ks[k], zk, -vk, Bk_diag])
            zmin = all_vals.min()
            zmax = all_vals.max()
            if zmax == zmin:
                zmax = zmin + 1e-10

            # 重新加密alpha_k用当前范围
            q_alpha = quantize1(alpha_ks[k], delta, zmin, zmax)
            alpha_hat = [encrypt(int(qi), pub) for qi in q_alpha]

            # 量化加密 zk, -vk, Bk_diag
            q_z   = quantize2(zk,     delta, zmin, zmax)
            q_v   = quantize2(-vk,    delta, zmin, zmax)
            q_B   = quantize2(Bk_diag, delta, zmin, zmax)

            c_z = [encrypt(int(qi), pub) for qi in q_z]
            c_v = [encrypt(int(qi), pub) for qi in q_v]

            # 同态计算
            x_hat_k = []
            for i in range(Nk):
                c_sum   = homo_add(c_z[i], c_v[i], pub)
                c_mul   = homo_mul_const(c_sum, int(q_B[i]), pub)
                c_final = homo_add(alpha_hat[i], c_mul, pub)
                x_hat_k.append(c_final)

            # 解密逆量化
            decrypted  = np.array([decrypt(c, priv) for c in x_hat_k], dtype=float)
            scale      = (zmax - zmin)**2 / delta**2
            correction = (2 * Bk_diag + (zk - (-vk)) + 1) * zmin - 2 * zmin**2
            xk         = decrypted * scale + correction
            x_new[k*Nk:(k+1)*Nk] = xk

        x = x_new
        z = soft_threshold(v + x, lam / rho)
        v = v + x - z

        mse = np.mean((A @ x - y) ** 2)
        mse_list.append(mse)

        if t % 10 == 0:
            print(f"迭代 {t}: MSE = {mse:.6f}")

        if mse < 1e-4:
            print(f"第 {t} 轮收敛")
            break

    return x, mse_list
