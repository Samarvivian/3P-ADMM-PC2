# 每次开机连接说明

## 节点信息

| 节点 | 角色 | hostname | Host | 端口（每次变化） |
|------|------|----------|------|----------------|
| 主节点 | Master · RTX A4000 | Z9kMJk | hz-t3.matpool.com | 见控制台 |
| edge1 | 边缘节点1 · RTX A2000 | ZBOj8J | hz-t3.matpool.com | 见控制台 |
| edge2 | 边缘节点2 · RTX A2000 | P6kMbo | hz-t3.matpool.com | 见控制台 |
| edge3 | 边缘节点3 · RTX A2000 | w0kmWy | hz-4.matpool.com | 见控制台 |

> 注意：每次重新开机后端口都会重新分配，需要去矩池云控制台查看最新端口。

---

## 每次开机操作流程

### 第一步：启动四台实例

去矩池云控制台，四台实例全部点开机，等待状态变成"运行中"。

### 第二步：记录新端口

点开每台实例详情页，记录SSH端口，填入下表：

| 节点 | Host | 本次端口 |
|------|------|---------|
| 主节点 | hz-t3.matpool.com | |
| edge1 | hz-t3.matpool.com | |
| edge2 | hz-t3.matpool.com | |
| edge3 | hz-4.matpool.com | |

### 第三步：MobaXterm连接四台

每台操作：Session → SSH → 填入host和新端口 → 输密码连上。

### 第四步：更新主节点SSH config

在主节点终端执行（替换端口号为本次实际端口）：

```bash
cat > ~/.ssh/config << EOF
Host edge1
    HostName hz-t3.matpool.com
    Port 新端口1
    User root

Host edge2
    HostName hz-t3.matpool.com
    Port 新端口2
    User root

Host edge3
    HostName hz-4.matpool.com
    Port 新端口3
    User root
EOF
```

### 第五步：重新配置免密登录

```bash
ssh-copy-id edge1
ssh-copy-id edge2
ssh-copy-id edge3
```

每条输一次密码，看到 `Number of key(s) added: 1` 或 `All keys were skipped` 都说明成功。

### 第六步：更新 config.py

```bash
cat > ~/config.py << EOF
NODES = {
    'edge1': {'host': 'hz-t3.matpool.com', 'port': 新端口1},
    'edge2': {'host': 'hz-t3.matpool.com', 'port': 新端口2},
    'edge3': {'host': 'hz-4.matpool.com',  'port': 新端口3},
}
EOF
```

### 第七步：验证连通

```bash
ssh edge1 "hostname"
ssh edge2 "hostname"
ssh edge3 "hostname"
```

三台分别返回 `ZBOj8J`、`P6kMbo`、`w0kmWy` 说明全部连通，可以开始写代码。

---

## 注意事项

- **代码和数据必须存在 `/mnt` 目录下**，否则关机后丢失
- 每次开机后记得把最新代码从GitHub拉取：`git pull`
- 长时间运行的任务用 `tmux` 或 `nohup` 挂到后台，防止SSH断开导致任务中断
- 用完记得去控制台**停止并释放**实例，避免继续计费

---

## 快速检查清单

每次开机后确认以下几项再开始工作：

- [ ] 四台实例全部运行中
- [ ] MobaXterm四个标签页全部连上
- [ ] `ssh edge1/edge2/edge3 "hostname"` 全部返回正确hostname
- [ ] `~/config.py` 端口已更新
- [ ] 代码已从GitHub拉取最新版本