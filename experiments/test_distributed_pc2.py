import numpy as np
import sys
sys.path.append('/mnt/3p-admm-pc2')
from protocol.master_node import run_distributed

# 读取节点配置
from config import NODES

np.random.seed(42)
M, N, K = 50, 99, 3
sparsity = 0.1

x_true = np.zeros(N)
idx = np.random.choice(N, int(N*sparsity), replace=False)
x_true[idx] = np.random.randn(int(N*sparsity))

A = np.random.randn(M, N) / np.sqrt(M)
y = A @ x_true + 0.01 * np.random.randn(M)

nodes = [
    {'host': NODES['edge1']['host'], 'port': NODES['edge1']['port']},
    {'host': NODES['edge2']['host'], 'port': NODES['edge2']['port']},
    {'host': NODES['edge3']['host'], 'port': NODES['edge3']['port']},
]

print("="*50)
print("运行分布式 3P-ADMM-PC2...")
_, mse_pc2 = run_distributed(A, y, nodes, K=K, rho=1.0, lam=0.05,
                               max_iter=100, delta=10**10, bits=1024)

print("="*50)
print(f"最终MSE: {mse_pc2[-1]:.6f}")
