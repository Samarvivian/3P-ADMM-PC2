# 节点连接配置（每次开机后更新端口）
NODES = {
    'edge1': {'host': 'hz-t3.matpool.com', 'port': 26535},
    'edge2': {'host': 'hz-t3.matpool.com', 'port': 29178},
    'edge3': {'host': 'hz-4.matpool.com',  'port': 28672},
}

# ADMM参数
RHO = 1.0       # 惩罚参数
LAMBDA = 1.0    # L1正则化参数
MAX_ITER = 100  # 最大迭代次数
TOL = 1e-4      # 收敛阈值

# 实验参数
M = 3000        # 观测维度
N = 27000       # 信号维度
K = 3           # 边缘节点数量
SPARSITY = 0.1  # 稀疏度
