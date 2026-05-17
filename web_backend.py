from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tempfile
import os
from typing import List
import threading
import uuid
from collections import defaultdict
from collections import deque
import threading
import builtins
import time

# 假设你的主流程在experiments/test_distributed_pc2.py或相关模块
from protocol.master_node import run_distributed
from config import NODES
import importlib
import importlib.util
import sys
import os
protocol_status = None
try:
    protocol_status = importlib.import_module("protocol.status")
except Exception:
    try:
        this_dir = os.path.dirname(__file__)
        status_path = os.path.join(this_dir, 'protocol', 'status.py')
        spec = importlib.util.spec_from_file_location('protocol.status', status_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules['protocol.status'] = module
        spec.loader.exec_module(module)
        protocol_status = module
    except Exception:
        protocol_status = None


# --- in-memory server log buffer (captures print calls in this process) ---
_log_buffer = deque(maxlen=2000)
_log_lock = threading.Lock()
_orig_print = builtins.print
def _print_wrapper(*args, **kwargs):
    try:
        s = ' '.join(str(a) for a in args)
    except Exception:
        try:
            s = str(args)
        except Exception:
            s = '<unserializable>'
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    entry = f'[{ts}] {s}'
    try:
        with _log_lock:
            _log_buffer.append(entry)
    except Exception:
        pass
    # still print to original stdout (force flush)
    try:
        kw = dict(kwargs)
        kw['flush'] = True
        _orig_print(*args, **kw)
    except Exception:
        pass

# patch builtins.print once
try:
    builtins.print = _print_wrapper
except Exception:
    pass

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
    # reset protocol status at the start of a new job (if available)
    if protocol_status is not None:
        try:
            protocol_status.reset()
        except Exception:
            pass

    # 保存临时上传文件 to disk and start background thread
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    job_id = uuid.uuid4().hex

    def _process_job(tmp_path_local, job_id_local, k_val, rho_val, lam_val, max_iter_val, delta_val, bits_val):
        try:
            # load data
            if file.filename.endswith('.npy'):
                data_local = np.load(tmp_path_local, allow_pickle=True)
            elif file.filename.endswith('.csv'):
                data_local = np.loadtxt(tmp_path_local, delimiter=',')
            else:
                JOB_RESULTS[job_id_local] = {"error": "仅支持npy/csv文件"}
                return

            # simple shape check
            if data_local.ndim == 1:
                y_local = data_local
                A_local = np.eye(len(y_local))
            elif data_local.ndim == 2:
                A_local = data_local[:, :-1]
                y_local = data_local[:, -1]
            else:
                JOB_RESULTS[job_id_local] = {"error": "数据格式错误"}
                return

            nodes_local = [
                {'name': 'edge1', 'host': NODES['edge1']['host'], 'port': NODES['edge1']['port']},
                {'name': 'edge2', 'host': NODES['edge2']['host'], 'port': NODES['edge2']['port']},
                {'name': 'edge3', 'host': NODES['edge3']['host'], 'port': NODES['edge3']['port']},
            ]

            try:
                _, mse_pc2_local, _, _ = run_distributed(A_local, y_local, nodes_local, K=k_val, rho=rho_val, lam=lam_val,
                                                         max_iter=max_iter_val, delta=delta_val, bits=bits_val)
                JOB_RESULTS[job_id_local] = {"mse_curve": list(map(float, mse_pc2_local)), "final_mse": float(mse_pc2_local[-1])}
            except Exception as e:
                print("后端异常：", e)
                JOB_RESULTS[job_id_local] = {"error": str(e)}
        finally:
            try:
                os.remove(tmp_path_local)
            except Exception:
                pass

    # ensure job results store exists
    try:
        JOB_RESULTS
    except NameError:
        JOB_RESULTS = {}

    th = threading.Thread(target=_process_job, args=(tmp_path, job_id, k, rho, lam, max_iter, delta, bits), daemon=True)
    th.start()

    return {"job_id": job_id}


@app.get('/api/status')
def get_status():
    if protocol_status is None:
        return {"stage": "unknown", "detail": "status module unavailable", "node_states": {}, "_debug": {"pid": os.getpid(), "protocol_status_id": None, "node_state_keys": []}}
    try:
        s = protocol_status.get_status()
        if 'node_states' not in s:
            s['node_states'] = {}
        # attach lightweight debug info to help troubleshooting
        try:
            proto_id = id(protocol_status)
            keys = list(s.get('node_states', {}).keys())
        except Exception:
            proto_id = None
            keys = []
        s['_debug'] = {"pid": os.getpid(), "protocol_status_id": proto_id, "node_state_keys": keys}
        return s
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get('/api/job/{job_id}')
def get_job(job_id: str):
    try:
        if 'JOB_RESULTS' in globals() and job_id in JOB_RESULTS:
            return JOB_RESULTS[job_id]
        else:
            return {"status": "running"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get('/api/status/inspect')
def inspect_status():
    """Diagnostic endpoint: returns sys.modules keys and protocol.status id (if available)."""
    try:
        mods = [k for k in sys.modules.keys() if k.startswith('protocol') or k.startswith('crypto')]
        proto_id = id(protocol_status) if protocol_status is not None else None
        return {"pid": os.getpid(), "protocol_status_id": proto_id, "modules": mods}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get('/api/logs')
def get_logs(n: int = 200):
    """Return the most recent server logs (n lines)."""
    try:
        with _log_lock:
            items = list(_log_buffer)[-int(n):]
        return {"lines": items, "count": len(items)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"msg": "无人机数据安全回传与分布式隐私计算平台后端已启动"}


@app.get('/ui')
def ui_index():
    """Serve the frontend index.html from the same origin to avoid file:// / CORS issues."""
    try:
        base = os.path.dirname(__file__)
        # index.html is in the project root; compute absolute path
        idx = os.path.join(base, '..', 'index.html')
        idx = os.path.abspath(idx)
        if not os.path.exists(idx):
            return JSONResponse({"error": "index.html not found", "path": idx}, status_code=404)
        return FileResponse(idx, media_type='text/html')
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/nodes")
def get_nodes():
    """Return configured edge nodes (host/port) so front-end can display real node IPs."""
    try:
        nodes = []
        for name, info in NODES.items():
            nodes.append({
                "name": name,
                "host": info.get('host'),
                "port": info.get('port'),
                "address": f"{info.get('host')}:{info.get('port')}"
            })
        return {"nodes": nodes}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
