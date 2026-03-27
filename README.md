# 每次开机连接说明

## 节点信息

| 节点 | 角色 | GPU | Host | 端口（每次变化） |
|------|------|-----|------|----------------|
| 主节点 | Master | RTX A4000 | hz-t3.matpool.com | 见控制台 |
| edge1 | 边缘节点1 | RTX A2000 | 见控制台（每次可能变） | 见控制台 |
| edge2 | 边缘节点2 | RTX A2000 | 见控制台（每次可能变） | 见控制台 |
| edge3 | 边缘节点3 | RTX A2000 | 见控制台（每次可能变） | 见控制台 |

> 重要：不只是端口会变，Host（hz-t3 还是 hz-4）每次开机也可能变，一定要以控制台显示为准。

---

## 每次开机操作流程

### 第一步：启动四台实例

去矩池云控制台，四台实例全部点开机，等待状态变成"运行中"。

### 第二步：记录四台的Host和端口

点开每台实例详情页 → SSH标签，记录Host和Port，填入下表：

| 节点 | Host | 本次端口 |
|------|------|---------|
| 主节点 | | |
| edge1 | | |
| edge2 | | |
| edge3 | | |

### 第三步：MobaXterm连接四台

每台操作：Session → SSH → 填入本次Host和Port → 输密码连上。

### 第四步：恢复SSH密钥软链接

密钥存在/mnt下不会丢失，但每次开机需要重新软链接。

在主节点执行：

```bash
ln -sf /mnt/ssh_keys/id_rsa ~/.ssh/id_rsa
ln -sf /mnt/ssh_keys/id_rsa.pub ~/.ssh/id_rsa.pub
```

验证密钥存在：

```bash
ls -la ~/.ssh/id_rsa
```

### 第五步：更新主节点SSH config

在主节点执行（用本次实际Host和端口替换）：

```bash
cat > ~/.ssh/config << EOF
Host edge1
    HostName 本次edge1的Host
    Port 本次edge1的端口
    User root

Host edge2
    HostName 本次edge2的Host
    Port 本次edge2的端口
    User root

Host edge3
    HostName 本次edge3的Host
    Port 本次edge3的端口
    User root
EOF
```

### 第六步：重新配置免密登录

```bash
ssh-copy-id edge1
ssh-copy-id edge2
ssh-copy-id edge3
```

每条输一次密码，看到以下任意一条都说明成功：
- `Number of key(s) added: 1`
- `All keys were skipped because they already exist`

### 第七步：更新 config.py

```bash
cat > /mnt/3p-admm-pc2/config.py << EOF
# 节点连接配置（每次开机后更新）
NODES = {
    'edge1': {'host': '本次edge1的Host', 'port': 本次edge1的端口},
    'edge2': {'host': '本次edge2的Host', 'port': 本次edge2的端口},
    'edge3': {'host': '本次edge3的Host', 'port': 本次edge3的端口},
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
EOF
```

### 第八步：验证连通

```bash
ssh edge1 "hostname"
ssh edge2 "hostname"
ssh edge3 "hostname"
```

三台各返回一个hostname说明全部连通，可以开始写代码。

---

## 注意事项

- **代码和数据必须存在 `/mnt` 目录下**，否则关机后丢失
- **SSH密钥存在 `/mnt/ssh_keys/`**，每次开机执行第四步软链接恢复
- **Host和端口每次都可能变**，不要用上次的，以控制台为准
- 长时间运行的任务用 `tmux` 挂到后台，防止SSH断开导致任务中断：

```bash
tmux new -s main      # 新建session
tmux attach -t main   # 重新连接session
```

- 用完记得去控制台**停止并释放**实例，避免继续计费

---

## 快速检查清单

每次开机后确认以下几项再开始工作：

- [ ] 四台实例全部运行中
- [ ] MobaXterm四个标签页全部连上
- [ ] `/mnt/ssh_keys/id_rsa` 软链接已恢复
- [ ] `~/.ssh/config` 已更新为本次Host和端口
- [ ] `ssh edge1/edge2/edge3 "hostname"` 全部返回正确hostname
- [ ] `/mnt/3p-admm-pc2/config.py` 端口已更新
