# 节点连接配置（每次开机后更新）
NODES = {
    'edge1': {'host': 'hz-4.matpool.com', 'port': 27920},
    'edge2': {'host': 'hz-4.matpool.com', 'port': 27304},
    'edge3': {'host': 'hz-4.matpool.com', 'port': 26688},
}

# ADMM参数
RHO = 1.0
LAMBDA = 1.0
MAX_ITER = 100
TOL = 1e-4

# 实验参数
M = 3000
N = 27000
K = 3
SPARSITY = 0.1
