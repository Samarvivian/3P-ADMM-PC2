import numpy as np
from numpy.linalg import inv
import sys
import pickle
import subprocess
import tempfile
import os
sys.path.append('/mnt/3p-admm-pc2')
from crypto.paillier import generate_keypair, encrypt, decrypt, homo_add, homo_mul_const
try:
    from crypto.paillier_gpu import encrypt_batch_gpu
    USE_GPU = True
    print('GPU加密已启用')
except Exception as e:
    USE_GPU = False
    print(f'GPU加密不可用，使用CPU: {e}')
from crypto.quantization import gamma1

def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def quantize2(v, delta, zmin, zmax):
    return np.floor(delta * (np.array(v) - zmin) / (zmax - zmin)).astype(object)

def quantize1(v, delta, zmin, zmax):
    return np.floor(delta**2 * (np.array(v) - zmin) / (zmax - zmin)**2).astype(object)

def send_to_edge(host, port, data, remote_path):
    """通过SSH把数据发送到边缘节点"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        pickle.dump(data, f)
        tmp_path = f.name
    subprocess.run([
        'scp', '-P', str(port), tmp_path,
        f'root@{host}:{remote_path}'
    ], check=True)
    os.unlink(tmp_path)

def recv_from_edge(host, port, remote_path, local_path):
    """从边缘节点接收数据"""
    subprocess.run([
        'scp', '-P', str(port),
        f'root@{host}:{remote_path}',
        local_path
    ], check=True)
    with open(local_path, 'rb') as f:
        return pickle.load(f)

def run_distributed(A, y, nodes, K=3, rho=1.0, lam=1.0,
                    max_iter=100, delta=10**10, bits=1024):
    """
    分布式3P-ADMM-PC2
    nodes: [{'host':..., 'port':...}, ...]
    """
    M, N = A.shape
    Nk = N // K
    A_blocks = [A[:, k*Nk:(k+1)*Nk] for k in range(K)]

    # 初始化
    print("【初始化】生成密钥...")
    pub, priv = generate_keypair(bits=bits)

    print("【初始化】预计算各节点矩阵...")
    B_blocks = []
    alpha_ks = []
    for k in range(K):
        Ak = A_blocks[k]
        Bk = inv(Ak.T @ Ak + rho * np.eye(Nk))
        B_blocks.append(Bk)
        alpha_ks.append(Bk @ (Ak.T @ y))

    # 量化范围
    all_vals = np.concatenate([alpha_ks[k] for k in range(K)] +
                               [np.diag(B_blocks[k]) for k in range(K)])
    ZMIN = all_vals.min() - abs(all_vals.min()) * 0.5
    ZMAX = all_vals.max() + abs(all_vals.max()) * 0.5

    # 数据安全共享阶段
    print("【数据安全共享】发送加密数据到边缘节点...")
    alpha_hats = []
    B_bar_qs = []
    for k in range(K):
        q_alpha = quantize1(alpha_ks[k], delta, ZMIN, ZMAX)
        if USE_GPU:
            alpha_hat = encrypt_batch_gpu([int(qi) for qi in q_alpha], pub)
        else:
            alpha_hat = [encrypt(int(qi), pub) for qi in q_alpha]
        alpha_hats.append(alpha_hat)

        q_B = quantize2(np.diag(B_blocks[k]), delta, ZMIN, ZMAX)
        B_bar_qs.append(q_B)

        # 发送到边缘节点
        data = {
            'pub': pub,
            'alpha_hat': alpha_hat,
            'B_bar_q': q_B,
            'delta': delta,
            'ZMIN': ZMIN,
            'ZMAX': ZMAX,
            'A_block': A_blocks[k],
        }
        send_to_edge(nodes[k]['host'], nodes[k]['port'],
                     data, f'/mnt/edge_data_k{k}.pkl')

    # 迭代
    print("【迭代计算】开始...")
    x = np.zeros(N)
    z = np.zeros(N)
    v = np.zeros(N)
    mse_list = []

    for t in range(max_iter):
        # 发送 zk, vk 给各边缘节点并触发计算
        for k in range(K):
            iter_data = {
                'zk': z[k*Nk:(k+1)*Nk],
                'vk': v[k*Nk:(k+1)*Nk],
                't': t,
            }
            send_to_edge(nodes[k]['host'], nodes[k]['port'],
                         iter_data, f'/mnt/iter_data_k{k}.pkl')

        # 触发边缘节点计算
        procs = []
        for k in range(K):
            cmd = (f"/root/miniconda3/envs/myconda/bin/python3 /mnt/3p-admm-pc2/protocol/edge_worker.py {k}")
            proc = subprocess.Popen([
                'ssh', '-p', str(nodes[k]['port']),
                f"root@{nodes[k]['host']}", cmd
            ])
            procs.append(proc)

        for proc in procs:
            proc.wait()

        # 收集结果并解密
        x_new = np.zeros(N)
        for k in range(K):
            result = recv_from_edge(
                nodes[k]['host'], nodes[k]['port'],
                f'/mnt/result_k{k}.pkl',
                f'/tmp/result_k{k}.pkl'
            )
            x_hat_k = result['x_hat_k']
            zk = z[k*Nk:(k+1)*Nk]
            vk = v[k*Nk:(k+1)*Nk]
            Bk_diag = np.diag(B_blocks[k])

            decrypted  = np.array([decrypt(c, priv) for c in x_hat_k], dtype=float)
            scale      = (ZMAX - ZMIN)**2 / delta**2
            correction = (2 * Bk_diag + (zk - (-vk)) + 1) * ZMIN - 2 * ZMIN**2
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
