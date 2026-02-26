# 盲图像处理系统 (Blind Image Processing)

基于同态加密（Paillier）的隐私保护图像处理系统，实现了客户端-服务器架构下的盲处理功能。

## 项目简介

本项目实现了一个隐私保护的图像处理系统，使用 Paillier 同态加密算法，允许服务器在不知道图像内容的情况下对加密图像进行处理（增亮、模糊等操作）。

**核心特性：**
- 🔐 完全隐私保护：服务器无法看到原始图像内容
- 🚀 支持多种操作：图像增亮、均值模糊
- 🌐 客户端-服务器架构：本地加密，远程处理
- 🔑 密钥安全：私钥仅保存在客户端

## 项目结构

```
.
├── master.py                    # 客户端主程序
├── test1.py                     # 本地测试脚本
├── step0_test.py                # 基础功能测试
├── parallel_test1.py            # 并行处理测试
├── blind_filter_server/         # 服务器端代码
│   ├── app.py                   # Flask 服务器
│   ├── requirements.txt         # 服务器依赖
│   └── render.yaml              # Render 部署配置
└── Image/                       # 测试图像目录
```

## 快速开始

### 环境要求

- Python 3.8+
- pip

### 安装依赖

```bash
# 客户端依赖
pip install phe pillow numpy requests

# 服务器端依赖（如需本地部署）
cd blind_filter_server
pip install -r requirements.txt
```

### 使用方法

#### 1. 本地测试（无需服务器）

```bash
python test1.py
```

这将在本地演示完整的加密-处理-解密流程。

#### 2. 客户端-服务器模式

修改 [master.py](master.py) 中的服务器 URL：

```python
SERVER_URL = "https://your-server-url.com"
```

运行客户端：

```bash
python master.py
```

#### 3. 部署服务器

**本地部署：**

```bash
cd blind_filter_server
python app.py
```

服务器将在 `http://localhost:10000` 启动。

**云端部署（Render）：**

项目已配置 [render.yaml](blind_filter_server/render.yaml)，可直接部署到 Render 平台。

## 工作原理

1. **客户端加密**：使用 Paillier 公钥加密图像的每个像素
2. **服务器盲处理**：服务器对加密数据进行同态运算（加法、乘法）
3. **客户端解密**：使用私钥解密处理后的结果

```
[客户端]                    [服务器]
  图像
   ↓
 加密 (公钥)
   ↓
 密文 ────────────→         密文
                              ↓
                          盲处理（增亮/模糊）
                              ↓
 密文 ←──────────────      处理后密文
   ↓
 解密 (私钥)
   ↓
处理后图像
```

## 支持的操作

### 增亮 (Brighten)

```python
run('Image/1.png', operation='brighten')
```

对图像进行增亮处理，服务器通过同态加法实现。

### 模糊 (Blur)

```python
run('Image/1.png', operation='blur')
```

对图像进行均值模糊处理，服务器通过同态加法计算邻域平均值。

## API 接口

### POST /blind_process

**请求参数：**

```json
{
  "public_key_n": "公钥 n 值（字符串）",
  "encrypted_pixels": ["加密像素数组"],
  "operation": "brighten 或 blur",
  "amount": 80,
  "width": 16,
  "height": 16
}
```

**响应：**

```json
{
  "result": ["处理后的加密像素数组"],
  "counts": [1, 1, ...],
  "status": "ok"
}
```

## 性能说明

- 加密/解密速度取决于密钥长度（默认 1024 位）
- 建议测试时使用小尺寸图像（16x16 或 32x32）
- 大图像处理时间较长，可考虑分块处理

## 安全性

- 使用 Paillier 同态加密算法，提供语义安全性
- 私钥永不离开客户端
- 服务器仅能看到加密后的大整数，无法推断原始图像内容

## 技术栈

- **加密库**：python-paillier
- **图像处理**：Pillow, NumPy
- **服务器框架**：Flask
- **HTTP 客户端**：requests

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
