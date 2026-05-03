# 3P-ADMM-PC2：分布式隐私计算框架复现

参考论文《Parallel Collaborative ADMM Privacy Computing and Adaptive GPU Acceleration for Distributed Edge Networks》https://ieeexplore.ieee.org/document/11340748，实现了一个基于Paillier同态加密的分布式ADMM隐私计算框架和GPU加速方案。
该项目已获得全国第十届密码技术竞赛三等奖。
---

## 目录

- [项目背景](#项目背景)
- [硬件环境](#硬件环境)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [算法原理](#算法原理)
- [实验结果](#实验结果)
- [与论文的差距分析](#与论文的差距分析)
- [注意事项](#注意事项)

---

## 项目背景

本项目复现了3P-ADMM-PC2算法，该算法将分布式ADMM与Paillier同态加密结合，实现了在不泄露数据隐私的前提下完成分布式LASSO问题求解。核心思想是：

- 主节点持有观测向量y和压缩矩阵A，将A按列分块分发给K个边缘节点
- 边缘节点计算本地子问题，通过同态加密保护中间结果
- 主节点完成全局z和v的更新，迭代收敛

整个算法分三个阶段：初始化阶段、数据安全共享阶段、并行隐私计算阶段。

---

## 硬件环境

| 节点 | 硬件 | 说明 |
|------|------|------|
| 主节点 | RTX A4000 GPU | 负责加密、解密、z/v更新 |
| 边缘节点×3 | RTX A2000 GPU | 负责矩阵求逆、同态计算 |

---

## 项目结构

```
3p-admm-pc2/
├── admm/
│   ├── centralized.py          # 集中式ADMM（基准对比，需大内存）
│   ├── distributed.py          # 标准分布式ADMM（无加密，对比基准）
│   └── dp_admm.py              # 差分隐私ADMM（DP-ADMM对比）
│
├── crypto/
│   ├── paillier.py             # Paillier同态加密核心
│   │                           #   - 密钥生成（generate_keypair）
│   │                           #   - 加解密（encrypt/decrypt）
│   │                           #   - 同态加法（homo_add）
│   │                           #   - 同态常数乘法（homo_mul_const）
│   └── paillier_gpu.py         # GPU加速批量加密
│                               #   - encrypt_batch_gpu: GPU算g^m，CPU算r^n
│                               #   - encrypt_batch_gpu_fast: 预计算r^n（9240 OPS）
│                               #   - encrypt_batch_gpu_crt: 分布式CRT版本
│
├── gpu/
│   ├── cufft_modexp.cu         # cuFFT全GPU版ModExp（最优，10056 OPS）
│   │                           #   使用Barrett Reduction + cuFFT大整数乘法
│   └── reg_ntt_modexp_v3.cu    # 寄存器级NTT版ModExp（7260 OPS）
│
├── protocol/
│   ├── master_node.py          # 主节点完整协议
│   │                           #   - 密钥生成与分发
│   │                           #   - 分发Ak给边缘节点触发矩阵求逆
│   │                           #   - GPU批量加密alpha_k
│   │                           #   - 每轮加密z/v发给边缘节点
│   │                           #   - 收集密文解密反量化得到x
│   │                           #   - 更新z和v
│   ├── edge_worker.py          # 边缘节点迭代计算
│   │                           #   - 量化z和v并加密
│   │                           #   - 同态矩阵向量乘法：
│   │                           #     小规模（Nk≤100）：完整矩阵，精确
│   │                           #     大规模（Nk>100）：对角近似，高效
│   └── edge_init.py            # 边缘节点初始化
│                               #   - 接收Ak，计算Bk=(Ak^T*Ak+ρI)^{-1}
│                               #   - 计算alpha_k=Bk*Ak^T*y
│                               #   - 返回Bk和alpha_k给主节点
│
├── experiments/
│   ├── test_distributed_pc2.py # 小规模验证（M=50, N=99, K=3）
│   ├── test_large_scale.py     # 大规模实验（M=3000, N=27000, K=3）
│   ├── fig6_mse.png            # MSE收敛曲线（对应论文Fig.6）
│   └── results_large.npy       # 大规模实验数据
│
└── config.py                   # 节点SSH配置（每次开机后更新）
```

---

## 快速开始

### 1. 每次开机重新编译GPU库

`/tmp`目录不持久化，每次开机需重新编译：

```bash
nvcc -arch=sm_86 -O2 --compiler-options '-fPIC' \
    -dc gpu/cufft_modexp.cu -o /tmp/cufft_modexp.o

cat > /tmp/wr_cufft.cu << 'EOF'
#include <stdint.h>
extern "C" {
    void cufft_init(int N);
    void cufft_modexp(uint32_t*, uint32_t*, uint32_t*, uint32_t*, uint32_t*, int, int, int);
}
extern "C" {
void init_gpu(int N){ cufft_init(N); }
void run_modexp(uint32_t *hg, uint32_t *hm, uint32_t *hn, uint32_t *hR,
                uint32_t *ho, int N, int mb, int nb){
    cufft_modexp(hg,hm,hn,hR,ho,N,mb,nb);
}
}
EOF

nvcc -arch=sm_86 -O2 --compiler-options '-fPIC' -dc /tmp/wr_cufft.cu -o /tmp/wr_cufft.o
nvcc -arch=sm_86 --compiler-options '-fPIC' \
    -dlink /tmp/cufft_modexp.o /tmp/wr_cufft.o -o /tmp/dl_cufft.o
g++ -shared -fPIC /tmp/cufft_modexp.o /tmp/wr_cufft.o /tmp/dl_cufft.o \
    -lcuda -lcudart -lcufft -L/usr/local/cuda/lib64 -o /tmp/lib_cufft.so
```

### 2. 更新节点配置

每次开机后在矩池云控制台查看SSH地址和端口，更新`config.py`：

```python
NODES = {
    'edge1': {'host': 'xxx.matpool.com', 'port': 12345},
    'edge2': {'host': 'xxx.matpool.com', 'port': 12346},
    'edge3': {'host': 'xxx.matpool.com', 'port': 12347},
}
```

### 3. 同步代码到边缘节点

```bash
for NODE in "edge1 host1 port1" "edge2 host2 port2" "edge3 host3 port3"; do
    HOST=$(echo $NODE | cut -d' ' -f2)
    PORT=$(echo $NODE | cut -d' ' -f3)
    scp -P $PORT protocol/edge_worker.py protocol/edge_init.py \
        root@$HOST:/mnt/3p-admm-pc2/protocol/
done
```

### 4. 运行实验

小规模验证：
```bash
python3 experiments/test_distributed_pc2.py
```

大规模实验（后台运行）：
```bash
nohup python3 -u experiments/test_large_scale.py > /tmp/exp_log.txt 2>&1 &
tail -f /tmp/exp_log.txt
```

---

## 算法原理

### ADMM迭代更新

$$x_k^{(t)} = (A_k^T A_k + \rho I)^{-1}(A_k^T y + \rho(z_k^{(t-1)} - v_k^{(t-1)}))$$

$$z^{(t)} = S_{\lambda/\rho}(v^{(t-1)} + x^{(t)})$$

$$v^{(t)} = v^{(t-1)} + x^{(t)} - z^{(t)}$$

其中 $B_k = (A_k^T A_k + \rho I)^{-1}$，$\alpha_k = B_k A_k^T y$，则 $x_k = \alpha_k + B_k \rho(z_k - v_k)$。

### 量化方案（Theorem 1）

为支持Paillier加密（只支持非负整数），对实数进行量化：

```
Γ1(v) = floor(Δ² * (v - zmin) / (zmax - zmin)²)   # 用于alpha_k
Γ2(v) = floor(Δ  * (v - zmin) / (zmax - zmin))      # 用于Bk, z, v
```

本项目使用 Δ=10^10，ZMIN=-3.0，ZMAX=3.0。

### 反量化公式

小规模（完整矩阵，Nk≤100）：
```
correction = ZMIN + ZMIN*sum(zk-vk) + 2*ZMIN*B_rowsum - 2*ZMIN²*Nk
```

大规模（对角近似，Nk>100）：
```
correction = ZMIN * (1 + 2*(B_diag - ZMIN) + (zk - vk))
```

### GPU加速ModExp（Algorithm 2）

将大整数模幂运算分解为多项式乘法，利用FFT加速：

1. 大整数表示为base=65536进制的向量（长度L=128）
2. 多项式乘法用cuFFT加速：O(L²) → O(L log L)
3. Barrett Reduction替代除法模运算
4. 批量任务并行处理

---

## 实验结果

### GPU加速ModExp性能（对应论文Table II，1024-bit密钥）

| 实现 | OPS | 相对CPU |
|------|-----|---------|
| CPU（gmpy2） | 9885 | 基准 |
| GPU串行v1 | 703 | 0.07x |
| GPU NTT版 | 1246 | 0.13x |
| GPU reg_ntt_v2 | 1589 | 0.16x |
| GPU libntt2 | 6578 | 0.67x |
| GPU reg_ntt_v3 | 7260 | 0.73x |
| **GPU cuFFT版** | **10056** | **1.02x（超过CPU）** |
| 论文GPU（cufftdx） | 79352 | 8x |

### GPU批量加密性能（EP）

| 方案 | OPS | 说明 |
|------|-----|------|
| CPU EP | 530 | 实时计算r^n |
| GPU EP（预计算r^n） | 9240 | 加速比17.4x |
| 论文GPU EP | 137807 | cufftdx，差距14.9x |

### MSE验证（对应论文Fig.6）

**小规模（M=50, N=99, K=3）：**

| 算法 | 最终MSE |
|------|---------|
| 标准Dis-ADMM | 0.111004 |
| 3P-ADMM-PC2 | 0.111003 |
| 差距 | **1e-6** |

**大规模（M=3000, N=27000, K=3，sparsity=10%，λ=1，ρ=1）：**

| 算法 | 最终MSE |
|------|---------|
| Dis-ADMM | 0.097331 |
| 3P-ADMM-PC2 | 0.098265 |
| 差距 | **9.34e-4** |

两者MSE曲线几乎重合，证明加密引入的误差可以忽略。

---

## 与论文的差距分析

| 指标 | 本项目 | 论文 | 原因 |
|------|--------|------|------|
| GPU ModExp | 10056 OPS | 79352 OPS | 缺少cufftdx（需NVIDIA授权） |
| GPU EP | 9240 OPS | 137807 OPS | 同上，差距约14.9倍 |
| MSE差距（大规模） | 9.34e-4 | 1e-14 | Δ=10^10 vs 论文Δ=10^15 |
| 边缘节点硬件 | RTX A2000 | 树莓派5 | 我们更强，延迟更低 |

**关于cufftdx：** 论文使用NVIDIA的cufftdx库（设备端FFT，数据全程在SM寄存器/shared memory中），避免了全局内存访问，比我们使用的全局cuFFT快约14倍。cufftdx需要从NVIDIA官网注册申请，本项目环境未安装。

**关于论文Table II的Edge Node GPU数据：** 经验证，论文Edge Node GPU（9729 OPS）实际是在主节点RTX 4060上模拟边缘节点算法测量的，树莓派5的VideoCore VII GPU不支持CUDA，无法运行该测试。

---

## 注意事项

1. **每次开机**需重新编译`/tmp/lib_cufft.so`，并更新`config.py`中的节点配置
2. **磁盘限制**：/mnt只有5GB，9000×9000的Bk矩阵每个619MB，实验后及时清理`/mnt/*.pkl`
3. **ZMIN/ZMAX**：必须覆盖迭代过程中z和v的实际范围，设为[-3, 3]；若缩小范围会导致clip截断误差累积发散
4. **大规模实验**：边缘节点负责矩阵求逆（9000×9000），主节点内存不足以处理
5. **对角近似**：大规模下同态矩阵乘法计算量为O(Nk²)不可行，使用对角近似；当A为高斯随机矩阵时非对角元素/对角元素≈0.37%，近似误差可忽略


# 致谢
该项目由本人历时五个月完成，前后历经不少挫折，在此由衷感谢提供idea和论文指导的师兄，也欢迎各位提出issue或pull request,为该项目的精进贡献力量！
