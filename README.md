---

# 🧩 3P-ADMM-PC2

### 基于 Paillier 同态加密的分布式 ADMM 算法实现

---

## 📘 1. 项目概述

**3P-ADMM-PC2 (Privacy-Preserving Parallel ADMM with Paillier & CuPy)**
旨在构建一个支持 **隐私保护 + 多节点协同 + GPU 加速** 的分布式优化框架。

系统目标：

* 💡 **隐私保护计算**：通过 Paillier 同态加密保障中间变量隐私；
* ⚙️ **并行加速**：利用 GPU / 多节点分布式并行机制；
* 🧠 **算法适配**：支持 LASSO、Logistic Regression 等典型稀疏优化问题；
* 🔬 **模块化设计**：可独立运行集中式版本或主从分布式版本。

---

## 🏗️ 2. 系统模块结构

| 模块名称                                         | 功能说明                                | 状态     |
|----------------------------------------------| ----------------------------------- | ------ |
| **Cen_ADMM_gpu.py**                          | 基于 CuPy 的集中式 ADMM（单机 GPU 版本，用作性能对照） | ✅ 已完成  |
| **master_node_priv_multi.py**                | 主节点（聚合 / 同态解密 / 参数广播）               | 🚧 开发中 |
| **client_node_ckks.py / paillier_client.py** | 从节点（局部加密计算 / 通信）                    | 🚧 开发中 |
| **utils_ckks.py / utils_paillier.py**        | 同态加密算法封装工具（CKKS / Paillier）         | 🚧 开发中 |
| **evaluation/**                              | 性能测试与实验结果保存目录                       | 📊 规划中 |

---

## ⚙️ 3. 环境要求

| 环境组件                   | 版本建议                              |
| ---------------------- | --------------------------------- |
| Python                 | ≥ 3.10                            |
| CUDA Toolkit           | ≥ 12.0                            |
| CuPy                   | `cupy-cuda12x`                    |
| NumPy                  | ≥ 1.26                            |
| Matplotlib             | ≥ 3.8                             |
| PyCryptodome / TenSEAL | 若使用同态加密功能                         |
| GPU                    | NVIDIA A100 / RTX / 其他支持 CUDA 的设备 |

---

## 🧩 4. 模块：Centralized ADMM (Baseline)

> **文件：** `centralized_admm_gpu.py`
> **功能：** 实现集中式 ADMM 求解 LASSO，用作后续主从式 / 加密式 ADMM 的性能对照实验。

主要特性：

* 使用 **CuPy** 在 GPU 上实现稀疏回归的 ADMM；
* 自动选择求解器（Cholesky / Conjugate Gradient）；
* 自动检测显存并自适应切换；
* 输出 MSE 收敛曲线与误差记录。

运行方式：

```bash
python Cen-ADMM.py
```

示例输出：

```
[INFO] A shape: 3000 x 27000, dtype=float32
[MEM] GPU free 39.85 GB, total 40.00 GB
[INFO] solver selected: direct
[ITER 100] MSE=8.407e-02 | solver=direct | time=0.024s | free=39.80GB
[Saved] Plot → Cen-ADMM-GPU_20251109_210301.png
```

参数说明：

| 参数            | 含义                                 |
| ------------- | ---------------------------------- |
| `M`, `N`      | 样本数与特征维度                           |
| `rho`         | ADMM 增广拉格朗日系数                      |
| `lamb`        | LASSO 正则化参数                        |
| `solver_pref` | `'auto'` / `'direct'` / `'cg'`     |
| `dtype`       | GPU 精度：`cp.float32` / `cp.float64` |

---

## 🚀 5. 开发进度规划

| 模块                             | 内容              | 阶段  |
| ------------------------------ | --------------- | --- |
| ✅ **Centralized Baseline**     | GPU 加速版集中式 ADMM | 已完成 |
| 🧩 **Master–Client Framework** | 通信、加密聚合、同步机制    | 进行中 |
| 🔐 **Paillier / CKKS 同态加密集成**  | 客户端局部加密与聚合      | 进行中 |
| 📊 **性能对比与论文复现**               | 含 GPU、加密、通信延迟分析 | 规划中 |

---




## 💬 6. 社区与交流
该方案由暨南大学**黄维维**、**刘浩**、**刘婷文**、**金灿灿**共同协作完成。

如果你在使用或阅读本项目的过程中发现问题、疑问或有新的想法，
欢迎通过以下方式参与交流与贡献：

* 🐞 **提交 Issue**：报告错误、提出改进建议或技术讨论；
* 🌟 **Star 本仓库**：如果本项目对你有帮助，请给一个 Star，
  这将是我们持续完善与开源更多隐私计算工具的最大动力；
* 🔄 **Pull Request**：欢迎贡献代码、优化实验、补充文档或扩展功能模块。

> 💡 我们相信开放与合作能让 3P-ADMM-PC2 走得更远。
> 期待你的反馈与参与，一起完善这一隐私保护分布式优化框架！

---



