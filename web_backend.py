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
import traceback
from typing import Any, Dict
import json

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


@app.post('/api/logs/clear')
def clear_logs():
    """Clear the in-memory server log buffer."""
    try:
        with _log_lock:
            _log_buffer.clear()
        print('[logs] cleared by API')
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post('/api/debug_gpu')
def debug_gpu(iters: int = 10, size: int = 9000):
    """Trigger a background GPU workload (encrypt_batch_gpu) repeatedly to make GPU utilization visible in slow samplers.
    Only use for local debugging. Returns immediately while job runs in background.
    """
    def _worker(i, s):
        try:
            print(f'[debug_gpu] start iterations={i} size={s}')
            # lazy import GPU function
            try:
                from crypto.paillier_gpu import encrypt_batch_gpu
                from crypto.paillier import generate_keypair
            except Exception as ex:
                print('[debug_gpu] GPU module import failed:', ex)
                return
            pub, priv = generate_keypair(bits=1024)
            messages = list(range(s))
            for _ in range(i):
                encrypt_batch_gpu(messages, pub)
            print('[debug_gpu] finished')
        except Exception:
            print('[debug_gpu] error', traceback.format_exc())

    th = threading.Thread(target=_worker, args=(iters, size), daemon=True)
    th.start()
    return {"status": "started", "iters": iters, "size": size}

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


# --- GPU metrics endpoint ---
# in-memory deque to hold recent GPU samples posted by monitor
# each entry: { 'ts': float, 'index': int, 'utilization': float, 'memory_used_mb': float, 'memory_total_mb': float }
_gpu_samples = deque(maxlen=1200)  # default keep last 1200 samples (~10 minutes at 0.5s)
_gpu_lock = threading.Lock()

# in-memory deque for system samples (cpu, ram)
# each entry: { 'ts': float, 'cpu_util': float, 'ram_used_gb': float, 'ram_total_gb': float }
_system_samples = deque(maxlen=1200)
_system_lock = threading.Lock()


@app.post('/api/gpu_sample')
def post_gpu_sample(sample: Dict[str, Any]):
    """Receive a single GPU sample from experiments/monitor_gpu.py and store in a rolling in-memory deque.

    Expected JSON shape (one of these):
      {"ts": 12345.6, "gpus": [{"index":0, "utilization":10.0, "memory_used_mb":100, "memory_total_mb":4000}, ...]}
    or normalized single-gpu sample:
      {"ts": 12345.6, "index": 0, "utilization": 10.0, "memory_used_mb": 100, "memory_total_mb": 4000}
    The endpoint will store one entry per-GPU (flattened) so front-end can request recent samples.
    """
    try:
        # accept both batched 'gpus' or single-gpu flat formats
        ts = float(sample.get('ts', time.time()))
        entries = []
        if 'gpus' in sample and isinstance(sample['gpus'], list):
            for g in sample['gpus']:
                try:
                    entries.append({
                        'ts': ts,
                        'index': int(g.get('index', 0)),
                        'utilization': float(g.get('utilization', 0.0)),
                        'memory_used_mb': float(g.get('memory_used_mb', g.get('memory_used', 0.0))),
                        'memory_total_mb': float(g.get('memory_total_mb', g.get('memory_total', 0.0))),
                    })
                except Exception:
                    continue
        else:
            # flat sample
            try:
                entries.append({
                    'ts': ts,
                    'index': int(sample.get('index', 0)),
                    'utilization': float(sample.get('utilization', 0.0)),
                    'memory_used_mb': float(sample.get('memory_used_mb', sample.get('memory_used', 0.0))),
                    'memory_total_mb': float(sample.get('memory_total_mb', sample.get('memory_total', 0.0))),
                })
            except Exception:
                pass

        if entries:
            with _gpu_lock:
                for e in entries:
                    _gpu_samples.append(e)
            try:
                print(f'[metrics] received gpu samples added={len(entries)} newest_ts={entries[-1].get("ts")}, index={entries[-1].get("index")}')
            except Exception:
                pass
        return {"status": "ok", "added": len(entries)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post('/api/system_sample')
def post_system_sample(sample: Dict[str, Any]):
    """Receive a single system sample from monitor and store in-memory deque.

    Expected JSON shape: {"ts": 12345.6, "cpu_util": 12.3, "ram_used_gb": 3.2, "ram_total_gb": 16.0}
    """
    try:
        ts = float(sample.get('ts', time.time()))
        cpu = float(sample.get('cpu_util', sample.get('cpu', 0.0)))
        ram_used = float(sample.get('ram_used_gb', sample.get('ram_used', 0.0)))
        ram_total = float(sample.get('ram_total_gb', sample.get('ram_total', 0.0)))
        entry = {'ts': ts, 'cpu_util': cpu, 'ram_used_gb': ram_used, 'ram_total_gb': ram_total}
        with _system_lock:
            _system_samples.append(entry)
        try:
            print(f'[metrics] received system sample ts={entry.get("ts")}, cpu={entry.get("cpu_util")}, ram={entry.get("ram_used_gb")}')
        except Exception:
            pass
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get('/api/system_metrics')
def system_metrics(n: int = 300):
    """Return recent system samples from in-memory deque.

    Returns: {"metrics": [{"ts": <sec>, "cpu_util": <float>, "ram_used_gb": <float>, "ram_total_gb": <float>}, ...]}
    """
    try:
        with _system_lock:
            items = list(_system_samples)
        return {"metrics": items[-int(n):]}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get('/api/metrics_debug')
def metrics_debug():
    """Return counts and last sample for GPU and system metrics (diagnostic)."""
    try:
        with _gpu_lock:
            gcount = len(_gpu_samples)
            glut = _gpu_samples[-1] if gcount else None
        with _system_lock:
            scount = len(_system_samples)
            slut = _system_samples[-1] if scount else None
        return {"gpu_count": gcount, "last_gpu": glut, "system_count": scount, "last_system": slut}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get('/api/gpu_metrics')
def gpu_metrics(n: int = 300, gpu_index: int = 0):
    """Return recent GPU metrics from the in-memory deque, optionally filtered to one GPU index.

    Returns: {"metrics": [{"ts": <sec>, "utilization": <float>, "memory_used_mb": <float>, "memory_total_mb": <float>, "index": <int>} ...]}
    """
    try:
        with _gpu_lock:
            items = list(_gpu_samples)
        if gpu_index is not None:
            items = [i for i in items if i.get('index', 0) == int(gpu_index)]
        return {"metrics": items[-int(n):]}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
