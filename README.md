# 3P-ADMM-PC2-GUI

基于 ADMM 算法的稀疏信号重建系统，支持分布式、加密分布式两种模式，提供图形界面。

## 使用方式

### 方式一：打包成 exe 运行

```powershell
# 安装依赖
pip install -r requirements.txt

# 打包
.\build_windows.bat
```
打包完成后默认放在 `dist/3P-ADMM-PC2.exe` ，双击即可运行

### 方式二：从源码运行

```powershell
# 安装依赖
pip install -r requirements.txt

# 运行
python src/main.py
```

### GPU 加速（可选）：有 NVIDIA GPU 的用户可安装 CuPy，程序会自动检测并使用。
```
pip install cupy-cuda12x
```

## 运行模式
- **分布式**：多台电脑通过局域网协同计算，一台运行 Master，其余运行 Edge
- **加密分布式**：分布式基础上加入 Paillier 同态加密保护数据隐私

## 分布式使用步骤

1. Master 电脑启动程序，选择 Master 模式，记下邀请码和 IP
2. Edge 电脑启动程序，选择 Edge 模式，填入 Master 的 IP 和邀请码，点击加入
3. Master 加载数据，分配列数，点击开始计算

## 项目结构

```
src/
├── main.py              # 入口
├── gui/                 # 界面
├── core/                # ADMM 算法、加密、量化
├── api/edge_server.py   # Edge 节点 Flask 服务
├── network/client.py    # Master→Edge HTTP 通信
└── utils/gpu.py         # GPU 加速（CuPy/NumPy 自动切换）
```

