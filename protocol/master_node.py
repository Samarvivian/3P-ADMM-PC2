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
                    max_iter=100, delta=10**10, bits=1024, x_true=None):
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

    print("【初始化】发送Ak给边缘节点，由边缘节点计算Bk...")
    for k in range(K):
        init_data = {'Ak': A_blocks[k], 'y': y, 'rho': rho}
        send_to_edge(nodes[k]['host'], nodes[k]['port'],
                     init_data, f'/mnt/init_data_k{k}.pkl')
    procs = []
    for k in range(K):
        cmd = f"/root/miniconda3/envs/myconda/bin/python3 /mnt/3p-admm-pc2/protocol/edge_init.py {k}"
        proc = subprocess.Popen(['ssh', '-p', str(nodes[k]['port']),
            f"root@{nodes[k]['host']}", cmd])
        procs.append(proc)
    for proc in procs:
        proc.wait()
    B_blocks = []
    alpha_ks = []
    for k in range(K):
        result = recv_from_edge(nodes[k]['host'], nodes[k]['port'],
            f'/mnt/init_result_k{k}.pkl', f'/tmp/init_result_k{k}.pkl')
        B_blocks.append(result['Bk'])
        alpha_ks.append(result['alpha_k'])
        print(f'  边缘节点{k} Bk计算完成')

    # 量化范围
    all_vals = np.concatenate([alpha_ks[k] for k in range(K)] +
                               [np.diag(B_blocks[k]) for k in range(K)])
    ZMIN = -3.0
    ZMAX = 3.0

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

        B_bar_qs.append(np.diag(B_blocks[k]))  # 保留对角用于反量化

        # 发送到边缘节点（包含完整Bk矩阵）
        data = {
            'pub': pub,
            'alpha_hat': alpha_hat,
            'B_k': B_blocks[k] if Nk <= 100 else np.diag(B_blocks[k]),  # 小规模完整，大规模对角
            'rho': rho,
            'delta': delta,
            'ZMIN': ZMIN,
            'ZMAX': ZMAX,
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
                'zk': rho * z[k*Nk:(k+1)*Nk],  # 乘以rho
                'vk': rho * v[k*Nk:(k+1)*Nk],  # 乘以rho
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
            zk = rho * z[k*Nk:(k+1)*Nk]  # 和发送给边缘节点的一致
            vk = rho * v[k*Nk:(k+1)*Nk]  # 和发送给边缘节点的一致
            Bk_diag = np.diag(B_blocks[k])

            decrypted  = np.array([decrypt(c, priv) for c in x_hat_k], dtype=float)
            scale      = (ZMAX - ZMIN)**2 / delta**2
            # 反量化公式（推导自Theorem 1）：
            # correction = ZMIN + ZMIN*sum(zk-vk) + 2*ZMIN*B_rowsum - 2*ZMIN^2*Nk
            Nk_local = len(zk)
            if Nk_local <= 100:
                # 小规模：完整矩阵，correction用行和
                B_row_sum = B_blocks[k].sum(axis=1)
                zv_sum = np.sum(zk - vk)
                correction = ZMIN + ZMIN*zv_sum + 2*ZMIN*B_row_sum - 2*ZMIN**2*Nk_local
            else:
                # 大规模：对角近似，correction逐元素
                B_diag = np.diag(B_blocks[k])
                correction = ZMIN*(1 + 2*(B_diag-ZMIN) + (zk-vk))
            xk         = decrypted * scale + correction
            x_new[k*Nk:(k+1)*Nk] = xk

        x = x_new
        z = soft_threshold(v + x, lam / rho)
        v = v + x - z

        if x_true is not None:
            mse = np.mean((x - x_true) ** 2)
        else:
            mse = np.mean((A @ x - y) ** 2)
        mse_list.append(mse)

        if t % 10 == 0:
            print(f"迭代 {t}: MSE = {mse:.6f}")

        if mse < 1e-4:
            print(f"第 {t} 轮收敛")
            break

    return x, mse_list, B_blocks, alpha_ks
