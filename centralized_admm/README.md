# Centralized ADMM (LASSO) Example

这是一个简单的集中式 ADMM 实现，用于求解 LASSO 问题：

    minimize (1/2)||X w - y||_2^2 + lambda * ||w||_1

实现要点：
- 使用变量分裂 w = z，把 L1 正则放到 z 上，采用 ADMM 进行求解（w-update, z-update, u-update）。
- 在仿真中生成稀疏真实参数 w_true，记录每次迭代的 MSE（z 与 w_true 之间）并绘图。

运行方法：

1. 安装依赖（建议使用虚拟环境）：

```powershell
python -m pip install -r requirements.txt
```

2. 运行脚本：

```powershell
python d:\开源\code\centralized_admm\admm_centralized.py
```

输出：控制台会打印运行的迭代次数与最终 MSE，并在当前目录生成 `mse_iterations.png`。

可修改脚本中的参数（样本数 n、特征数 p、正则参数 lambda、rho、最大迭代次数）以复现论文中的更多实验。
