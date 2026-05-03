import numpy as np
import sys, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append('/mnt/3p-admm-pc2')
from protocol.master_node import run_distributed
from config import NODES

print('='*50)
print('开始大规模实验')
print('='*50)

np.random.seed(42)
M, N, K = 3000, 27000, 3
sparsity = 0.1

print(f'\n[1/5] 生成数据 A∈R^{{{M}x{N}}}...')
t0 = time.time()
x_true = np.zeros(N)
idx = np.random.choice(N, int(N*sparsity), replace=False)
x_true[idx] = np.random.randn(int(N*sparsity))
A = np.random.randn(M, N) / np.sqrt(M)
y = A @ x_true + 0.01 * np.random.randn(M)
print(f'    数据生成完成: {time.time()-t0:.1f}s')
print(f'    A形状: {A.shape}, 稀疏度: {sparsity}')
sys.stdout.flush()

nodes = [
    {'host': NODES['edge1']['host'], 'port': NODES['edge1']['port']},
    {'host': NODES['edge2']['host'], 'port': NODES['edge2']['port']},
    {'host': NODES['edge3']['host'], 'port': NODES['edge3']['port']},
]
print(f'\n[2/5] 节点配置:')
for i,n in enumerate(nodes):
    print(f'    edge{i}: {n["host"]}:{n["port"]}')
sys.stdout.flush()

max_iter = 100
rho = 1.0
lam = 1.0

results = {}

print(f'\n[3/5] 运行 3P-ADMM-PC2 (max_iter={max_iter})...')
sys.stdout.flush()
t0 = time.time()
_, mse_pc2, B_blocks, alpha_ks = run_distributed(
    A, y, nodes, K=K, rho=rho, lam=lam,
    max_iter=max_iter, delta=10**10, bits=1024, x_true=x_true)
t_pc2 = time.time()-t0
print(f'    3P-ADMM-PC2完成!')
print(f'    总时间: {t_pc2:.0f}s = {t_pc2/3600:.2f}h')
print(f'    最终MSE: {mse_pc2[-1]:.6f}')
results['3P-ADMM-PC2'] = mse_pc2
sys.stdout.flush()

print(f'\n[4/5] 运行 Dis-ADMM（复用Bk，不重复求逆）...')
sys.stdout.flush()
Nk = N // K
A_blocks = [A[:, k*Nk:(k+1)*Nk] for k in range(K)]
alpha_ks_dis = [B_blocks[k] @ (A_blocks[k].T @ y) for k in range(K)]
print(f'    alpha_k计算完成')
sys.stdout.flush()

def soft_threshold(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0)

x = np.zeros(N); z = np.zeros(N); v = np.zeros(N)
mse_dis = []
t0 = time.time()
for t in range(max_iter):
    x_new = np.zeros(N)
    for k in range(K):
        xk = alpha_ks_dis[k] + B_blocks[k] @ (rho*(z[k*Nk:(k+1)*Nk]-v[k*Nk:(k+1)*Nk]))
        x_new[k*Nk:(k+1)*Nk] = xk
    x = x_new
    z = soft_threshold(v+x, lam/rho)
    v = v+x-z
    mse = np.mean((x-x_true)**2)
    mse_dis.append(mse)
    if t % 10 == 0:
        print(f'    迭代{t:3d}: MSE={mse:.6f}  ({time.time()-t0:.0f}s)')
        sys.stdout.flush()

t_dis = time.time()-t0
print(f'    Dis-ADMM完成! 总时间: {t_dis:.0f}s, 最终MSE={mse_dis[-1]:.6f}')
results['Dis-ADMM'] = mse_dis
sys.stdout.flush()

print(f'\n[5/5] 保存结果和画图...')
np.save('/mnt/3p-admm-pc2/experiments/results_large.npy', results)

plt.figure(figsize=(8,5))
colors = {'Dis-ADMM': 'blue', '3P-ADMM-PC2': 'red'}
for name, mse in results.items():
    plt.semilogy(mse, label=name, color=colors.get(name))
plt.xlabel('Number of Iterations')
plt.ylabel('MSE')
plt.title(f'MSE Comparison (A in R^{{{M}x{N}}})')
plt.legend()
plt.grid(True)
plt.savefig('/mnt/3p-admm-pc2/experiments/fig6_mse.png', dpi=150, bbox_inches='tight')
print('    图已保存: experiments/fig6_mse.png')

print('\n' + '='*50)
print('实验完成！最终结果:')
for name, mse in results.items():
    print(f'  {name}: {mse[-1]:.6f}')
print(f'  MSE差距: {abs(mse_pc2[-1]-mse_dis[-1]):.2e}')
print('='*50)

import os
import sys
sys.path.append('/mnt/3p-admm-pc2')
from config import NODES

# 关闭所有边缘节点
import subprocess
for name, node in NODES.items():
    print(f'关闭 {name}...')
    subprocess.run(['ssh', '-p', str(node['port']),
                   f'root@{node["host"]}',
                   'shutdown -h now'], timeout=10)

# 关闭主节点
print('关闭主节点...')
os.system('shutdown -h now')
