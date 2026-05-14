from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tempfile
import os
from typing import List

# 假设你的主流程在experiments/test_distributed_pc2.py或相关模块
from protocol.master_node import run_distributed
from config import NODES

app = FastAPI(title="无人机数据安全回传与分布式隐私计算平台")

# 允许跨域，方便前端本地开发
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/upload_and_process/")
async def upload_and_process(
    file: UploadFile = File(...),
    k: int = Form(3),
    rho: float = Form(1.0),
    lam: float = Form(0.05),
    max_iter: int = Form(100),
    delta: int = Form(10**10),
    bits: int = Form(1024)
):
    """
    接收无人机上传的特征数据文件（CSV/NPY），并进行分布式隐私计算。
    """
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # 读取数据（假设为npy或csv）
    try:
        if file.filename.endswith('.npy'):
            data = np.load(tmp_path, allow_pickle=True)
        elif file.filename.endswith('.csv'):
            data = np.loadtxt(tmp_path, delimiter=',')
        else:
            return JSONResponse({"error": "仅支持npy/csv文件"}, status_code=400)
    finally:
        os.remove(tmp_path)

    # 简单判断数据格式
    if data.ndim == 1:
        y = data
        A = np.eye(len(y))
    elif data.ndim == 2:
        A = data[:, :-1]
        y = data[:, -1]
    else:
        return JSONResponse({"error": "数据格式错误"}, status_code=400)

    # 构造节点信息
    nodes = [
        {'host': NODES['edge1']['host'], 'port': NODES['edge1']['port']},
        {'host': NODES['edge2']['host'], 'port': NODES['edge2']['port']},
        {'host': NODES['edge3']['host'], 'port': NODES['edge3']['port']},
    ]

    # 调用分布式隐私计算主流程
    try:
        _, mse_pc2, _, _ = run_distributed(A, y, nodes, K=k, rho=rho, lam=lam,
                                           max_iter=max_iter, delta=delta, bits=bits)
    except Exception as e:
        print("后端异常：", e)
        return JSONResponse({"error": str(e)}, status_code=500)

    return {"mse_curve": list(map(float, mse_pc2)), "final_mse": float(mse_pc2[-1])}

@app.get("/")
def root():
    return {"msg": "无人机数据安全回传与分布式隐私计算平台后端已启动"}
