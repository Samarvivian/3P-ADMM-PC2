# 节点连接配置（每次开机后更新端口）
NODES = {
    'edge1': {'host': 'hz-t3.matpool.com', 'port': 26471},
    'edge2': {'host': 'hz-t3.matpool.com', 'port': 27799},
    'edge3': {'host': 'hz-t3.matpool.com', 'port': 29459},
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
