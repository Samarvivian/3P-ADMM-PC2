"""
实时GPU+CPU+内存监控脚本（只采集，不跑实验）
用法：
  step1: python3 experiments/monitor_gpu.py &  # 后台启动监控
  step2: python3 experiments/test_distributed_pc2.py  # 跑实验
  step3: Ctrl+C停止监控
"""
import subprocess, time, csv, os, threading, signal
import psutil

def collect_system_data(output_file, stop_event, interval=0.5):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'gpu_util', 'gpu_mem_used_mb',
                        'cpu_util', 'ram_used_gb', 'ram_total_gb'])
        while not stop_event.is_set():
            try:
                result = subprocess.run(
                    ['nvidia-smi',
                     '--query-gpu=utilization.gpu,memory.used',
                     '--format=csv,noheader,nounits'],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                gpu_line = result.stdout.strip().split(',')
                gpu_util = float(gpu_line[0].strip())
                gpu_mem = float(gpu_line[1].strip())
                cpu_util = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory()
                ram_used = ram.used / 1024**3
                ram_total = ram.total / 1024**3
                ts = time.strftime('%Y/%m/%d %H:%M:%S')
                writer.writerow([ts, gpu_util, gpu_mem, cpu_util, ram_used, ram_total])
                f.flush()

                # POST到后端（如果配置了）
                try:
                    backend_url = os.environ.get('GPU_MONITOR_BACKEND',
                                                  'http://127.0.0.1:8000').rstrip('/')
                    import urllib.request, json as _json
                    gpu_record = {'ts': time.time(), 'gpus': [{
                        'index': 0, 'utilization': gpu_util,
                        'memory_used_mb': gpu_mem}]}
                    req = urllib.request.Request(
                        backend_url + '/api/gpu_sample',
                        data=_json.dumps(gpu_record).encode('utf-8'),
                        headers={'Content-Type': 'application/json'})
                    urllib.request.urlopen(req, timeout=1)
                    sys_record = {'ts': time.time(), 'cpu_util': cpu_util,
                                  'ram_used_gb': ram_used, 'ram_total_gb': ram_total}
                    req2 = urllib.request.Request(
                        backend_url + '/api/system_sample',
                        data=_json.dumps(sys_record).encode('utf-8'),
                        headers={'Content-Type': 'application/json'})
                    urllib.request.urlopen(req2, timeout=1)
                except Exception:
                    pass

            except Exception:
                pass
            time.sleep(interval)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=float, default=0.5)
    parser.add_argument('--out', default='/tmp/system_data.csv')
    parser.add_argument('--window', type=int, default=1200)
    args = parser.parse_args()

    stop_event = threading.Event()

    def handler(sig, frame):
        print('\n停止监控...')
        stop_event.set()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    print(f'系统监控已启动（间隔{args.interval}s，数据→{args.out}）')
    print('按Ctrl+C停止')

    collect_system_data(args.out, stop_event, args.interval)
    print('监控结束')
