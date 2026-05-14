import numpy as np

# 生成特征数据
np.random.seed(42)
M, N, K = 50, 99, 3
sparsity = 0.1

x_true = np.zeros(N)
idx = np.random.choice(N, int(N*sparsity), replace=False)
x_true[idx] = np.random.randn(int(N*sparsity))

A = np.random.randn(M, N) / np.sqrt(M)
y = A @ x_true + 0.01 * np.random.randn(M)

# 合并A和y，最后一列为y
features = np.hstack([A, y.reshape(-1, 1)])

# 保存为npy和csv
np.save('test/features.npy', features)
np.savetxt('test/features.csv', features, delimiter=',')

print('特征数据已生成并保存在 test/features.npy 和 test/features.csv')
