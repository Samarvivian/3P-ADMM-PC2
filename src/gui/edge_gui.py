"""
Edge节点 GUI

The Edge node:
  1. Starts a local Flask computation server (edge_server.py)
  2. Registers with the Master by sending its URL + invite code
  3. Waits for Master to push A_k and y, then computes local updates
  4. Displays task progress and system metrics
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import socket
import datetime
import time
import sys
import os
import logging

from gui.style import setup_style, decode_widget_tree
from gui import i18n
from gui import config as cfg

logger = logging.getLogger(__name__)


def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"



class EdgeWindow:
    """
    GUI for an Edge node.

    encrypted: whether to display encryption-related info
    """

    DEFAULT_EDGE_PORT = 9100
    DEFAULT_MASTER_PORT = 9000

    def __init__(self, root: tk.Tk, encrypted: bool = False):
        self.root = root
        self.encrypted = encrypted
        self._alive = True

        self._flask_thread: threading.Thread | None = None
        self._flask_app = None
        self._server_running = False
        self._registered = False
        self._edge_state = None

        mode_str = i18n.t("encrypted") if encrypted else i18n.t("distributed")
        self.root.title(f"3P-ADMM-PC2  [{mode_str}]  Edge")
        self.root.geometry("700x860")
        self.root.minsize(560, 600)
        self.root.resizable(True, True)
        setup_style(self.root)

        self._build_ui()
        decode_widget_tree(self.root)
        self._start_perf_monitor()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Outer frame holds the canvas + scrollbar
        outer = ttk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True)

        vscroll = ttk.Scrollbar(outer, orient=tk.VERTICAL)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._scroll_canvas = tk.Canvas(outer, highlightthickness=0,
                                        yscrollcommand=vscroll.set)
        self._scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vscroll.config(command=self._scroll_canvas.yview)

        # Inner frame is the real content area
        main = ttk.Frame(self._scroll_canvas, padding=12)
        self._scroll_win_id = self._scroll_canvas.create_window(
            (0, 0), window=main, anchor="nw")

        # Keep inner frame width in sync with canvas width
        def _on_canvas_resize(e):
            self._scroll_canvas.itemconfig(self._scroll_win_id, width=e.width)
        self._scroll_canvas.bind("<Configure>", _on_canvas_resize)

        # Update scroll region when content changes
        def _on_frame_resize(e):
            self._scroll_canvas.configure(
                scrollregion=self._scroll_canvas.bbox("all"))
        main.bind("<Configure>", _on_frame_resize)

        # Mouse-wheel scrolling
        def _on_mousewheel(e):
            self._scroll_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        self._scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ── Toolbar ───────────────────────────────────────────────────────
        toolbar = ttk.Frame(main)
        toolbar.pack(fill=tk.X, pady=(0, 10))

        color = "#8e44ad" if self.encrypted else "#27ae60"
        label = (i18n.t('encrypted') if self.encrypted else i18n.t('distributed'))
        tk.Label(toolbar, text=f"  {label}  {i18n.t('edge_node')}  ",
                 bg=color, fg="white", font=("Arial", 11, "bold"), padx=6, pady=2).pack(
            side=tk.LEFT)
        ttk.Button(toolbar, text="中/EN", width=6,
                   command=self._toggle_lang).pack(side=tk.RIGHT)

        # ── Local server panel ────────────────────────────────────────────
        srv_frame = ttk.LabelFrame(main, text=i18n.t("local_server"), padding=10)
        srv_frame.pack(fill=tk.X, pady=(0, 10))

        row1 = ttk.Frame(srv_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text=i18n.t("listen_ip"), width=14).pack(side=tk.LEFT)
        self.listen_ip_var = tk.StringVar(value=_get_local_ip())
        ttk.Entry(row1, textvariable=self.listen_ip_var, width=18).pack(side=tk.LEFT)

        row2 = ttk.Frame(srv_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text=i18n.t("port"), width=14).pack(side=tk.LEFT)
        self.port_var = tk.IntVar(value=self.DEFAULT_EDGE_PORT)
        ttk.Entry(row2, textvariable=self.port_var, width=8).pack(side=tk.LEFT)

        srv_btn_row = ttk.Frame(srv_frame)
        srv_btn_row.pack(fill=tk.X, pady=(6, 0))
        self.start_srv_btn = ttk.Button(srv_btn_row, text=i18n.t("start_server"),
                                        command=self.start_server)
        self.start_srv_btn.pack(side=tk.LEFT)
        self.stop_srv_btn = ttk.Button(srv_btn_row, text=i18n.t("stop_server"),
                                       command=self.stop_server, state=tk.DISABLED)
        self.stop_srv_btn.pack(side=tk.LEFT, padx=6)

        self.srv_status_var = tk.StringVar(value=i18n.t("server_stopped"))
        ttk.Label(srv_btn_row, textvariable=self.srv_status_var,
                  foreground="red").pack(side=tk.LEFT, padx=6)

        # ── Join Master panel ─────────────────────────────────────────────
        join_frame = ttk.LabelFrame(main, text=i18n.t("join_master"), padding=10)
        join_frame.pack(fill=tk.X, pady=(0, 10))

        row3 = ttk.Frame(join_frame)
        row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text=i18n.t("master_ip"), width=14).pack(side=tk.LEFT)
        self.master_ip_var = tk.StringVar(value="192.168.1.")
        ttk.Entry(row3, textvariable=self.master_ip_var, width=18).pack(side=tk.LEFT)

        row4 = ttk.Frame(join_frame)
        row4.pack(fill=tk.X, pady=2)
        ttk.Label(row4, text=i18n.t("master_port"), width=14).pack(side=tk.LEFT)
        self.master_port_var = tk.IntVar(value=self.DEFAULT_MASTER_PORT)
        ttk.Entry(row4, textvariable=self.master_port_var, width=8).pack(side=tk.LEFT)

        row5 = ttk.Frame(join_frame)
        row5.pack(fill=tk.X, pady=2)
        ttk.Label(row5, text=i18n.t("invite_code_label"), width=14).pack(side=tk.LEFT)
        self.token_var = tk.StringVar()
        ttk.Entry(row5, textvariable=self.token_var,
                  width=10, font=("Courier", 13, "bold")).pack(side=tk.LEFT)

        join_btn_row = ttk.Frame(join_frame)
        join_btn_row.pack(fill=tk.X, pady=(6, 0))
        self.join_btn = ttk.Button(join_btn_row, text=i18n.t("join"),
                                   command=self.join_master, state=tk.DISABLED)
        self.join_btn.pack(side=tk.LEFT)
        self.leave_btn = ttk.Button(join_btn_row, text=i18n.t("leave"),
                                    command=self.leave_master, state=tk.DISABLED)
        self.leave_btn.pack(side=tk.LEFT, padx=6)

        self.join_status_var = tk.StringVar(value=i18n.t("not_registered"))
        ttk.Label(join_btn_row, textvariable=self.join_status_var,
                  foreground="gray").pack(side=tk.LEFT, padx=6)

        ttk.Label(join_frame, text=i18n.t("start_first"),
                  foreground="gray", font=("Arial", 8)).pack(anchor=tk.W, pady=(4, 0))

        # ── Task status panel ─────────────────────────────────────────────
        task_frame = ttk.LabelFrame(main, text=i18n.t("task_status"), padding=10)
        task_frame.pack(fill=tk.X, pady=(0, 10))

        self.task_status_var = tk.StringVar(value=i18n.t("idle"))
        ttk.Label(task_frame, textvariable=self.task_status_var).pack(anchor=tk.W)

        prog_row = ttk.Frame(task_frame)
        prog_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(prog_row, text=i18n.t("iteration")).pack(side=tk.LEFT)
        self.task_progress = ttk.Progressbar(prog_row, mode="determinate", maximum=100)
        self.task_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        # ── Performance panel ─────────────────────────────────────────────
        perf_frame = ttk.LabelFrame(main, text=i18n.t("performance"), padding=10)
        perf_frame.pack(fill=tk.X, pady=(0, 10))

        cpu_row = ttk.Frame(perf_frame)
        cpu_row.pack(fill=tk.X, pady=2)
        ttk.Label(cpu_row, text="CPU:", width=10).pack(side=tk.LEFT)
        self.cpu_bar = ttk.Progressbar(cpu_row, length=180, maximum=100)
        self.cpu_bar.pack(side=tk.LEFT)
        self.cpu_label = ttk.Label(cpu_row, text="0%", width=6)
        self.cpu_label.pack(side=tk.LEFT)

        mem_row = ttk.Frame(perf_frame)
        mem_row.pack(fill=tk.X, pady=2)
        ttk.Label(mem_row, text="Memory:", width=10).pack(side=tk.LEFT)
        self.mem_bar = ttk.Progressbar(mem_row, length=180, maximum=100)
        self.mem_bar.pack(side=tk.LEFT)
        self.mem_label = ttk.Label(mem_row, text="0%", width=6)
        self.mem_label.pack(side=tk.LEFT)

        gpu_row = ttk.Frame(perf_frame)
        gpu_row.pack(fill=tk.X, pady=2)
        ttk.Label(gpu_row, text=i18n.t("gpu_label"), width=10).pack(side=tk.LEFT)
        self.gpu_bar = ttk.Progressbar(gpu_row, length=180, maximum=100)
        self.gpu_bar.pack(side=tk.LEFT)
        self.gpu_label = ttk.Label(gpu_row, text="N/A", width=10)
        self.gpu_label.pack(side=tk.LEFT)

        # ── Log panel ─────────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(main, text=i18n.t("log"), padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_frame, height=10, state=tk.DISABLED,
                                bg="#0b1220", fg="#e5e7eb", insertbackground="#e5e7eb",
                                font=("Courier", 9), relief=tk.FLAT, highlightthickness=1,
                                highlightcolor="#233044", highlightbackground="#233044")
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(log_frame, orient=tk.VERTICAL,
                           command=self.log_text.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=sb.set)
        self.log_text.tag_config("ERR", foreground="#f44747")
        self.log_text.tag_config("OK",  foreground="#6a9955")

    # ── Language toggle ───────────────────────────────────────────────────────

    def _toggle_lang(self):
        new_lang = "en" if i18n.get_lang() == "zh" else "zh"
        i18n.set_lang(new_lang)
        cfg.set("lang", new_lang)
        self._refresh_labels()

    def _refresh_labels(self):
        listen_ip = self.listen_ip_var.get()
        port = self.port_var.get()
        master_ip = self.master_ip_var.get()
        master_port = self.master_port_var.get()
        token = self.token_var.get()
        join_status = self.join_status_var.get()
        srv_running = self._server_running
        registered = self._registered
        gpu_text = self.gpu_label.cget("text") if hasattr(self, "gpu_label") else "N/A"

        for w in self.root.winfo_children():
            w.destroy()

        self._build_ui()
        decode_widget_tree(self.root)

        self.listen_ip_var.set(listen_ip)
        self.port_var.set(port)
        self.master_ip_var.set(master_ip)
        self.master_port_var.set(master_port)
        self.token_var.set(token)
        self.join_status_var.set(join_status)
        self.gpu_label.config(text=gpu_text)

        if srv_running:
            self.start_srv_btn.config(state=tk.DISABLED)
            self.stop_srv_btn.config(state=tk.NORMAL)
            self.join_btn.config(state=tk.NORMAL if not registered else tk.DISABLED)
            self._set_srv_status(i18n.t("server_running"), "green")
        if registered:
            self.leave_btn.config(state=tk.NORMAL)
            self.join_btn.config(state=tk.DISABLED)

    # ── Server management ─────────────────────────────────────────────────────

    def start_server(self):
        if self._server_running:
            return
        try:
            port = int(self.port_var.get())
        except Exception:
            messagebox.showerror("Error", "Invalid port number")
            return

        from api.edge_server import EdgeState, create_app

        token = self.token_var.get().strip()
        state = EdgeState(session_token=token)
        self._edge_state = state
        app = create_app(state)
        self._flask_app = app

        import logging as _log
        for _name in ("werkzeug", "flask", "flask.app", "src.api.edge_server"):
            _log.getLogger(_name).setLevel(_log.WARNING)
        app.logger.setLevel(_log.WARNING)

        self._flask_thread = threading.Thread(
            target=lambda: app.run(host="0.0.0.0", port=port,
                                   use_reloader=False, use_debugger=False),
            daemon=True,
        )
        self._flask_thread.start()
        self._server_running = True

        self.start_srv_btn.config(state=tk.DISABLED)
        self.stop_srv_btn.config(state=tk.NORMAL)
        self.join_btn.config(state=tk.NORMAL)
        self.log(f"Server started on port {port}", tag="OK")
        self._set_srv_status(i18n.t("server_running"), "green")

        def _probe_gpu():
            from utils.gpu import is_gpu_available, get_xp, get_gpu_error, _find_system_cupy_paths
            if is_gpu_available():
                xp = get_xp()
                try:
                    dev = xp.cuda.Device()
                    name = dev.attributes.get("DeviceName", "GPU") if hasattr(dev, "attributes") else "GPU"
                except Exception:
                    name = "GPU"
                self.root.after(0, lambda: self.log(
                    f"{i18n.t('gpu_enabled')} — {name}", tag="OK"))
                self.root.after(0, lambda: self.gpu_label.config(text="0%"))
            else:
                err = get_gpu_error()
                msg = i18n.t("gpu_disabled") + (f" ({err})" if err else "")
                self.root.after(0, lambda m=msg: self.log(m))
                self.root.after(0, lambda: self.gpu_label.config(text="N/A"))
                # Show diagnostic: which paths were searched
                try:
                    paths = _find_system_cupy_paths()
                    if paths:
                        for p in paths:
                            import os
                            has_cupy = os.path.isdir(os.path.join(p, "cupy"))
                            self.root.after(0, lambda p=p, h=has_cupy: self.log(
                                f"  [GPU诊断] {'✓' if h else '✗'} {p}"))
                    else:
                        self.root.after(0, lambda: self.log("  [GPU诊断] 未找到任何Python安装路径"))
                except Exception as e:
                    self.root.after(0, lambda e=e: self.log(f"  [GPU诊断] 错误: {e}"))
        threading.Thread(target=_probe_gpu, daemon=True).start()

    def stop_server(self):
        self._server_running = False
        self._registered = False
        self.start_srv_btn.config(state=tk.NORMAL)
        self.stop_srv_btn.config(state=tk.DISABLED)
        self.join_btn.config(state=tk.DISABLED)
        self.leave_btn.config(state=tk.DISABLED)
        self._set_srv_status(i18n.t("server_stopped"), "red")
        self.join_status_var.set(i18n.t("not_registered"))
        self.log("Server stopped (process restart required for port reuse)")

    def _set_srv_status(self, text: str, color: str):
        self.srv_status_var.set(text)
        def _walk(w):
            try:
                if isinstance(w, ttk.Label) and w.cget("textvariable") == str(self.srv_status_var):
                    w.config(foreground=color)
                    return
            except Exception:
                pass
            for child in w.winfo_children():
                _walk(child)
        _walk(self.root)

    # ── Master registration ───────────────────────────────────────────────────

    def join_master(self):
        if not self._server_running:
            messagebox.showerror("Error", "Start the local server first.")
            return

        master_ip = self.master_ip_var.get().strip()
        master_port = self.master_port_var.get()
        token = self.token_var.get().strip()
        edge_port = self.port_var.get()
        edge_ip = self.listen_ip_var.get().strip()

        if not master_ip or not token:
            messagebox.showerror("Error", "Enter Master IP and Invite Code.")
            return

        master_url = f"http://{master_ip}:{master_port}"
        edge_url = f"http://{edge_ip}:{edge_port}"

        self.log(f"Registering with Master at {master_url}...")

        def _do_join():
            try:
                import requests
                resp = requests.post(
                    f"{master_url}/join",
                    json={"edge_url": edge_url, "token": token},
                    timeout=10,
                )
                data = resp.json()
                if data.get("ok"):
                    self.root.after(0, lambda: self._on_joined(master_url))
                else:
                    err = data.get("error", "Unknown error")
                    self.root.after(0, lambda: self._on_join_failed(err))
            except Exception as e:
                msg = str(e)
                self.root.after(0, lambda m=msg: self._on_join_failed(m))

        threading.Thread(target=_do_join, daemon=True).start()

    def _on_joined(self, master_url: str):
        self._registered = True
        self.join_btn.config(state=tk.DISABLED)
        self.leave_btn.config(state=tk.NORMAL)
        self.join_status_var.set(f"● Registered with {master_url}")
        self.log(f"Successfully registered with {master_url}", tag="OK")

    def _on_join_failed(self, error: str):
        self.log(f"Registration failed: {error}", tag="ERR")
        messagebox.showerror("Join Failed", f"Could not register with Master:\n{error}")

    def leave_master(self):
        self._registered = False
        self.join_btn.config(state=tk.NORMAL)
        self.leave_btn.config(state=tk.DISABLED)
        self.join_status_var.set(i18n.t("not_registered"))
        self.log("Left Master session")

    # ── Performance monitoring ────────────────────────────────────────────────

    def _start_perf_monitor(self):
        def _monitor():
            while self._alive:
                try:
                    import psutil
                    cpu = psutil.cpu_percent(interval=1)
                    mem = psutil.virtual_memory().percent
                    gpu_pct = None
                    gpu_mem_str = None
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_pct = util.gpu
                        used_mb = mem_info.used // (1024 * 1024)
                        total_mb = mem_info.total // (1024 * 1024)
                        gpu_mem_str = f"{gpu_pct}%  {used_mb}/{total_mb}MB"
                    except Exception:
                        pass
                    es = self._edge_state
                    count = es.compute_count if es else 0
                    max_it = es.max_iter if es else 0
                    busy = es.busy if es else False
                    if self._alive:
                        self.root.after(0, lambda c=cpu, m=mem, g=gpu_pct,
                                        gs=gpu_mem_str, n=count, mx=max_it, b=busy:
                                        self._update_perf_and_task(c, m, g, gs, n, mx, b))
                except Exception:
                    pass
                time.sleep(2)

        threading.Thread(target=_monitor, daemon=True).start()

    def _update_perf_and_task(self, cpu, mem, gpu_pct, gpu_mem_str,
                               compute_count, max_iter, busy):
        try:
            self.cpu_bar["value"] = cpu
            self.cpu_label.config(text=f"{cpu:.0f}%")
            self.mem_bar["value"] = mem
            self.mem_label.config(text=f"{mem:.0f}%")
            if gpu_pct is not None:
                self.gpu_bar["value"] = gpu_pct
                self.gpu_label.config(text=gpu_mem_str or f"{gpu_pct}%")
            else:
                self.gpu_bar["value"] = 0
            # Update task progress bar
            if max_iter > 0 and compute_count > 0:
                self.task_progress["maximum"] = max_iter
                self.task_progress["value"] = min(compute_count, max_iter)
                status = i18n.t("computing") if busy else i18n.t("idle")
                self.task_status_var.set(f"{status}  (iter {compute_count}/{max_iter})")
            elif compute_count == 0:
                self.task_progress["value"] = 0
                self.task_status_var.set(i18n.t("idle"))
        except tk.TclError:
            pass

    # ── Logging ───────────────────────────────────────────────────────────────

    def log(self, msg: str, tag: str = ""):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, line, tag)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        logger.info(msg)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_close(self):
        self._alive = False
        self.root.destroy()
