"""
3P-ADMM-PC2 启动器
"""
import subprocess, sys, os, time, webbrowser

def main():
    print("=" * 50)
    print("  3P-ADMM-PC2 分布式隐私计算平台")
    print("=" * 50)
    print("正在启动后端服务...")

    project_dir = '/mnt/3p-admm-pc2'
    python = '/root/miniconda3/envs/myconda/bin/python3'
    uvicorn = '/root/miniconda3/envs/myconda/bin/uvicorn'

    backend = subprocess.Popen(
        [uvicorn, 'web_backend:app', '--host', '0.0.0.0', '--port', '8000'],
        cwd=project_dir
    )

    time.sleep(3)
    print("服务已启动！")
    print("请在浏览器访问: http://localhost:8000/ui")
    print("或通过SSH隧道转发后在本地访问")
    print("按Ctrl+C停止服务")

    try:
        backend.wait()
    except KeyboardInterrupt:
        backend.terminate()
        print("服务已停止")

if __name__ == '__main__':
    main()
