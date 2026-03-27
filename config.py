# 节点连接配置（每次开机后更新）
NODES = {
    'edge1': {'host': '本次edge1的Host', 'port': 26610},
    'edge2': {'host': '本次edge2的Host', 'port': 26942},
    'edge3': {'host': '本次edge3的Host', 'port': 27274},
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
