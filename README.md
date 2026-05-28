# 3P-ADMM-PC2：分布式隐私计算框架

> 基于Paillier半同态加密与分布式ADMM算法的三阶段并行隐私计算框架，结合CUDA GPU加速技术，支持大规模数据的高效分布式优化。
>
> **该项目已获得第十届全国密码技术竞赛 国家三等奖**

---

## 目录

- [项目背景](#项目背景)
- [硬件环境](#硬件环境)
- [项目结构](#项目结构)
- [环境依赖](#环境依赖)
- [快速开始](#快速开始)
  - [1. 编译GPU库](#1-每次开机重新编译gpu库)
  - [2. 更新节点配置](#2-更新节点配置)
  - [3. 同步代码到边缘节点](#3-同步代码到边缘节点)
  - [4. 运行实验](#4-运行实验)
  - [5. 启动Web可视化平台](#5-启动web可视化平台)
  - [6. 使用可执行文件](#6-使用可执行文件)
- [算法原理](#算法原理)
- [实验结果](#实验结果)
- [注意事项](#注意事项)
- [项目署名](#项目署名)

---

## 项目背景

本项目实现了一种名为3P-ADMM-PC2的算法，将分布式ADMM与Paillier同态加密结合，实现在不泄露数据隐私的前提下完成分布式LASSO问题求解。

**三阶段框架：**

- **阶段一（初始化）**：主节点生成Paillier密钥对，分发公钥与数据矩阵$A_k$；边缘节点预计算$B_k=(A_k^TA_k+\rho I)^{-1}$和$\alpha_k$
- **阶段二（数据安全共享）**：主节点GPU批量加密当前迭代的$z$和$v$，通过SSH安全信道发送给各边缘节点
- **阶段三（并行隐私计算）**：边缘节点在密文域完成同态矩阵向量乘法，返回加密的$x_k$；主节点解密聚合，更新$z$和$v$，循环迭代至收敛

---

## 硬件环境

| 节点 | 硬件 | 说明 |
|------|------|------|
| 主节点 | NVIDIA RTX A4000 16GB | 负责加密、解密、z/v更新 |
| 边缘节点×3 | NVIDIA RTX A2000 12GB | 负责矩阵求逆、同态计算 |

> 本项目分布式实验环境基于[矩池云](https://matpool.com)搭建，主节点与三台边缘节点均为矩池云GPU实例，通过矩池云提供的内网SSH互联，模拟真实的分布式边缘计算网络拓扑。

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
│   │                           #   密钥格式：(lam, mu, n, n2, p, q)
│   │                           #   - generate_keypair：密钥生成
│   │                           #   - encrypt/decrypt：加解密
│   │                           #   - homo_add：同态加法
│   │                           #   - homo_mul_const：同态常数乘法
│   └── paillier_gpu.py         # GPU加速批量加密
│                               #   - precompute_rn：预计算r^n（两批奇偶轮交替）
│                               #   - encrypt_batch_gpu：GPU算g^m，CPU算r^n
│                               #   - encrypt_batch_gpu_fast：预计算r^n（9240 OPS）
│
├── gpu/
│   ├── cufft_modexp.cu         # cuFFT全GPU版ModExp（最优，10056 OPS）
│   │                           #   Barrett Reduction + cuFFT大整数多项式乘法
│   │                           #   base=65536，L=128维，gridDim=任务数，blockDim=32
│   └── reg_ntt_modexp_v3.cu    # 寄存器级NTT版ModExp（7260 OPS）
│                               #   warp内__shfl_xor_sync寄存器共享，零延迟
│
├── protocol/
│   ├── master_node.py          # 主节点完整协议
│   │                           #   - 密钥生成与公钥分发
│   │                           #   - GPU批量加密alpha_k、z、v
│   │                           #   - 奇偶轮交替r^n复用策略
│   │                           #   - proc.wait()同步等待所有节点
│   │                           #   - 解密聚合，soft_threshold更新z/v
│   ├── edge_worker.py          # 边缘节点迭代计算
│   │                           #   - 接收主节点加密的z/v（密文）
│   │                           #   - 同态矩阵向量乘法：
│   │                           #     小规模（Nk≤100）：完整矩阵，精确
│   │                           #     大规模（Nk>100）：对角近似，O(Nk)
│   │                           #   - 返回加密的x_k
│   └── edge_init.py            # 边缘节点初始化
│                               #   - 接收Ak，计算Bk=(Ak^TAk+ρI)^{-1}
│                               #   - 计算并量化alpha_k=Bk*Ak^T*y
│
├── experiments/
│   ├── test_distributed_pc2.py # 小规模验证（M=50, N=99, K=3）
│   ├── test_large_scale.py     # 大规模实验（M=3000, N=27000, K=3）
│   ├── monitor_gpu.py          # GPU/CPU/内存资源采集（独立运行，不跑实验）
│   ├── fig6_mse.png            # MSE收敛曲线
│   └── results_large.npy       # 大规模实验结果数据
│
├── static/
│   └── drone_bg.jpg            # Web登录页背景图
│
├── web_backend.py              # FastAPI后端
│                               #   路由：/ui  /performance  /api/status
│                               #         /api/logs  /api/nodes
│                               #         /api/gpu_metrics  /api/system_metrics
│                               #         /api/upload_and_process/
│
├── index.html                  # 前端主界面（登录/上传/实时监控/图表导出）
├── performance.html            # 性能对比分析独立页面
├── launcher.py                 # PyInstaller打包启动器入口
├── dist/
│   └── 3P-ADMM-PC2             # Linux可执行文件
└── config.py                   # 节点SSH配置（每次开机后更新）
```

---

## 环境依赖

```bash
# Python环境（推荐conda）
conda create -n myconda python=3.8
conda activate myconda

# 核心依赖
pip install gmpy2 numpy fastapi uvicorn psutil pycryptodome
```

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.8+ | 推荐conda环境 |
| CUDA | 12.1 | 编译架构sm_86（A4000/A2000） |
| gmpy2 | 最新 | CPU大整数运算基准 |
| FastAPI + uvicorn | 最新 | Web后端服务 |
| numpy | 最新 | 数值计算 |
| psutil | 最新 | 系统资源监控 |
| Chart.js | 4.x | 前端可视化（CDN自动引入） |

---

本项目一共有3个重要分支：
| 分支 | 说明 |
|------|------|
| \`features\` | 当前稳定版本，含Web可视化平台与可编译脚本 |
| \`releases\` | Web可视化应用发布版本 |
| \`master\` | 早期版本，存在中间变量未加密的漏洞，虽不适用使用但仍具有借鉴意义 |


## 快速开始

### 1. 每次开机重新编译GPU库

> `/tmp`目录不持久化，每次开机需重新编译：

```bash
# Step 1：编译主kernel
nvcc -arch=sm_86 -O2 --compiler-options '-fPIC' \
    -dc /mnt/3p-admm-pc2/gpu/cufft_modexp.cu -o /tmp/cufft_modexp.o

# Step 2：生成wrapper
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

# Step 3：编译wrapper
nvcc -arch=sm_86 -O2 --compiler-options '-fPIC' \
    -dc /tmp/wr_cufft.cu -o /tmp/wr_cufft.o

# Step 4：设备链接
nvcc -arch=sm_86 --compiler-options '-fPIC' \
    -dlink /tmp/cufft_modexp.o /tmp/wr_cufft.o -o /tmp/dl_cufft.o

# Step 5：生成共享库
g++ -shared -fPIC \
    /tmp/cufft_modexp.o /tmp/wr_cufft.o /tmp/dl_cufft.o \
    -lcuda -lcudart -lcufft -L/usr/local/cuda/lib64 \
    -o /tmp/lib_cufft.so

# 验证编译成功
ls -lh /tmp/lib_cufft.so
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
# 分别替换为实际的host和port
scp -P <port1> protocol/edge_worker.py protocol/edge_init.py \
    crypto/paillier.py config.py \
    root@<host1>:/mnt/3p-admm-pc2/protocol/

scp -P <port2> protocol/edge_worker.py protocol/edge_init.py \
    crypto/paillier.py config.py \
    root@<host2>:/mnt/3p-admm-pc2/protocol/

scp -P <port3> protocol/edge_worker.py protocol/edge_init.py \
    crypto/paillier.py config.py \
    root@<host3>:/mnt/3p-admm-pc2/protocol/
```

### 4. 运行实验

**小规模验证：**

```bash
cd /mnt/3p-admm-pc2
python3 experiments/test_distributed_pc2.py
# 预期：最终MSE ≈ 0.111003，与Dis-ADMM差距 < 1e-6
```

**大规模实验（M=3000, N=27000, K=3）：**

```bash
cd /mnt/3p-admm-pc2
nohup python3 -u experiments/test_large_scale.py > /tmp/exp_log.txt 2>&1 &
tail -f /tmp/exp_log.txt
# 预期：最终MSE ≈ 0.098265，与Dis-ADMM差距 < 9.34e-4
```

### 5. 启动Web可视化平台

**终端1：启动后端**

```bash
cd /mnt/3p-admm-pc2
uvicorn web_backend:app --host 0.0.0.0 --port 8000
```

**终端2：启动资源监控采集**

```bash
cd /mnt/3p-admm-pc2
GPU_MONITOR_BACKEND=http://127.0.0.1:8000 \
    python3 experiments/monitor_gpu.py --interval 0.5 &
```

**本地访问（SSH隧道）：**

```bash
# 在本地终端执行建立隧道
ssh -p <矩池云端口> -NL 8000:localhost:8000 root@<矩池云地址>

# 浏览器访问
# 主界面（登录/上传/实时监控）
http://localhost:8000/ui

# 性能对比分析页
http://localhost:8000/performance
```

### 6. 使用可执行文件

项目提供基于PyInstaller打包的Linux可执行文件，无需配置Python环境：

```bash
# 添加执行权限
chmod +x dist/3P-ADMM-PC2

# 直接运行（内置FastAPI后端，启动后访问 http://localhost:8000/ui）
./dist/3P-ADMM-PC2
```

> **注意**：可执行文件内置后端服务，但GPU共享库仍需按照步骤1单独编译。

---

## 算法原理

### ADMM迭代更新

$$x_k^{(t)} = B_k\left(\alpha_k + \rho(z_k^{(t-1)} - v_k^{(t-1)})\right)$$

$$z^{(t)} = S_{\lambda/\rho}\left(v^{(t-1)} + x^{(t)}\right)$$

$$v^{(t)} = v^{(t-1)} + x^{(t)} - z^{(t)}$$

其中 $B_k = (A_k^T A_k + \rho I)^{-1}$，$\alpha_k = B_k A_k^T y$。

### 量化方案

Paillier加密只支持非负整数，对实数进行线性量化（$\Delta=10^{10}$，$[z_{min}, z_{max}]=[-3, 3]$）：

$$\Gamma_1(v) = \left\lfloor \frac{\Delta^2 \cdot (v - z_{min})}{(z_{max} - z_{min})^2} \right\rfloor \quad \text{（用于 } \alpha_k\text{）}$$

$$\Gamma_2(v) = \left\lfloor \frac{\Delta \cdot (v - z_{min})}{z_{max} - z_{min}} \right\rfloor \quad \text{（用于 } B_k,\ z,\ v\text{）}$$

量化精度约 $6\times10^{-10}$，误差可忽略不计。

### GPU加速ModExp

将 $g^m \mod n^2$ 分解为多项式卷积，利用cuFFT加速：

1. 大整数表示为 base=65536 进制的128维向量
2. 多项式乘法用cuFFT加速：$O(N^2) \to O(N\log N)$
3. Barrett Reduction替代GPU不支持的大整数除法
4. 9000个任务→9000个block并行，每block 32线程（一个warp）协作
5. warp内通过 `__shfl_xor_sync` 零延迟寄存器共享完成NTT蝶形运算
6. 预计算两批 $r^n$，奇偶轮交替复用，端到端加速17.4×

---

## 实验结果

### GPU ModExp 各版本性能

| 版本 | OPS | 说明 |
|------|-----|------|
| CPU（gmpy2） | 9885 | 基准，GMP汇编级优化 |
| GPU 串行 v1 | 703 | Local Memory瓶颈 |
| GPU NTT版 | 1246 | O(NlogN)多项式乘法 |
| GPU reg_ntt_v3 | 7260 | warp寄存器共享，零延迟 |
| **GPU cuFFT版** | **10056** | **首次超越CPU（1.02×）** |
| **GPU + 预计算r^n** | **9240** | **端到端加速17.4×** |

### MSE验证

| 规模 | Dis-ADMM MSE | 3P-ADMM-PC2 MSE | 差距 |
|------|-------------|-----------------|------|
| 小规模（M=50, N=99） | 0.111004 | 0.111003 | **< 10⁻⁶** |
| 大规模（M=3000, N=27000） | 0.097331 | 0.098265 | **9.34×10⁻⁴** |

---

## 注意事项

1. **每次开机**需重新编译 `/tmp/lib_cufft.so`，并更新 `config.py` 节点配置
2. **磁盘限制**：`/mnt` 仅5GB，9000×9000的 $B_k$ 矩阵每个约619MB，实验后及时清理 `/mnt/*.pkl`
3. **ZMIN/ZMAX**：固定设为[-3, 3]，缩小范围会导致clip截断误差累积发散
4. **大规模实验**：$B_k$ 矩阵求逆（9000×9000）由边缘节点完成，主节点内存不足以处理
5. **对角近似**：大规模下 $N_k>100$ 时启用，将同态乘法从 $O(N_k^2)$ 降至 $O(N_k)$
6. **r^n复用**：两批奇偶轮交替策略在工程上缓解了密文复用问题，生产环境建议每轮使用新鲜随机数
7. **同步机制**：采用同步ADMM，慢节点会拖慢整体进度（木桶效应）

---

## 项目署名

| 项目名称 | 3P-ADMM-PC2 |
|----------|-------------|
| 作者 | Weiwei Huang, Cancan Jin, Hao Liu, Tingwen Liu, Donghong Cai |
| 单位 | 暨南大学网络空间安全学院（College of Cyber Security, Jinan University） |
| 竞赛 | 第十届全国密码技术竞赛 国家三等奖 |
