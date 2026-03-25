"""
Master节点 GUI

Background-1 scenario:
  - Master holds full A and y (measured locally)
  - Master splits A column-wise and pushes A_k to each Edge over the network
  - Edge nodes compute B_k / alpha_k / Bbar_k from their received A_k and y

Modes:
  distributed  — multi-machine, plain communication
  encrypted    — multi-machine, Paillier homomorphic encryption

Layout (left | right):
  Left  : A + y upload · ADMM params · session token + edge nodes + column split
  Right : Convergence plot · Result summary · Export
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import socket
import random
import string
import datetime
import os
import sys
import platform
import logging
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib

from gui.style import setup_style, decode_widget_tree
from gui import i18n
from gui import config as cfg

logger = logging.getLogger(__name__)


def _decode_unicode(text: str) -> str:
    return text


# ── Matplotlib font setup ─────────────────────────────────────────────────────
def _setup_fonts():
    s = platform.system()
    if s == "Windows":
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei UI", "Microsoft YaHei", "SimHei"]
    elif s == "Darwin":
        matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC"]
    else:
        matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

_setup_fonts()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _gen_token(length: int = 6) -> str:
    return "".join(random.choices(string.digits, k=length))


def _load_array(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    elif ext == ".mat":
        import scipy.io
        mat = scipy.io.loadmat(path)
        keys = [k for k in mat if not k.startswith("__")]
        if len(keys) == 1:
            return np.array(mat[keys[0]], dtype=float)
        raise ValueError(f".mat file has multiple variables: {keys}. "
                         "Rename the variable to 'y'.")
    elif ext == ".csv":
        return np.loadtxt(path, delimiter=",")
    elif ext in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            if len(keys) == 1:
                return f[keys[0]][:]
            raise ValueError(f".h5 file has multiple datasets: {keys}.")
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# ── Master registration server ────────────────────────────────────────────────

class MasterRegistrationServer:
    """
    Tiny Flask server that Edge nodes call to register themselves.

    POST /join   { "edge_url": "http://ip:port", "token": "XXXXXX" }
    GET  /nodes  → list of registered edge_urls
    """

    def __init__(self, token: str, port: int = 9000):
        self.token = token
        self.port = port
        self.edge_urls: list = []
        self._app = None
        self._thread = None
        self._on_join = None
        self._on_token_mismatch = None

    def set_on_join(self, cb):
        self._on_join = cb

    def set_on_token_mismatch(self, cb):
        self._on_token_mismatch = cb

    def start(self):
        from flask import Flask, request, jsonify
        import logging as _log
        _log.getLogger("werkzeug").setLevel(_log.WARNING)
        _log.getLogger("flask.app").setLevel(_log.WARNING)

        app = Flask("master_reg")
        app.logger.setLevel(logging.WARNING)

        edge_urls = self.edge_urls
        on_join = lambda url: self._on_join(url) if self._on_join else None
        on_mismatch = lambda exp, got: self._on_token_mismatch(exp, got) if self._on_token_mismatch else None

        @app.route("/join", methods=["POST"])
        def join():
            data = request.get_json() or {}
            received = data.get("token", "")
            current_token = self.token
            if received != current_token:
                logger.warning(f"[join] token mismatch: expected={current_token!r} received={received!r}")
                on_mismatch(current_token, received)
                return jsonify({"ok": False, "error": "Invalid token"}), 403
            url = data.get("edge_url", "").strip()
            if not url:
                return jsonify({"ok": False, "error": "edge_url required"}), 400
            if url not in edge_urls:
                edge_urls.append(url)
                on_join(url)
            return jsonify({"ok": True})

        @app.route("/nodes", methods=["GET"])
        def nodes():
            return jsonify({"ok": True, "nodes": edge_urls})

        self._app = app
        self._thread = threading.Thread(
            target=lambda: app.run(host="0.0.0.0", port=self.port,
                                   use_reloader=False, use_debugger=False),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self.edge_urls.clear()


# ── MasterWindow ──────────────────────────────────────────────────────────────

class MasterWindow:
    """
    Main window for the Master role.

    mode = "distributed"  → multi-machine, plain
    mode = "encrypted"    → multi-machine, Paillier encryption
    """

    MODES = ("distributed", "encrypted")
    REG_PORT = 9000

    def __init__(self, root: tk.Tk, mode: str = "distributed"):
        assert mode in self.MODES
        self.root = root
        self.mode = mode
        self.encrypted = (mode == "encrypted")

        # ── State ──────────────────────────────────────────────────────────
        self.A: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.A_path = tk.StringVar()
        self.y_path = tk.StringVar()

        self.rho_var = tk.DoubleVar(value=1.0)
        self.lamb_var = tk.DoubleVar(value=0.1)
        self.max_iter_var = tk.IntVar(value=100)
        self.tol_var = tk.DoubleVar(value=1e-6)

        self.result_x: np.ndarray | None = None
        self.residual_history: list = []
        self._running = False
        self._stop_flag = False
        self._alive = True

        self.session_token = _gen_token()
        self.reg_server: MasterRegistrationServer | None = None
        self.edge_urls: list = []
        # Per-edge column count vars: {url: tk.IntVar}
        self.col_vars: dict = {}

        # ── Build UI ───────────────────────────────────────────────────────
        mode_label = {"distributed": "Distributed", "encrypted": "Encrypted"}[mode]
        self.root.title(f"3P-ADMM-PC2  [{mode_label}]  Master")
        self.root.geometry("1280x800")
        self.root.minsize(1060, 680)
        setup_style(self.root)

        self._build_ui()
        decode_widget_tree(self.root)
        self._start_reg_server()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        toolbar = ttk.Frame(self.root, padding=(14, 8))
        toolbar.pack(fill=tk.X)

        mode_colors = {"distributed": "#27ae60", "encrypted": "#8e44ad"}
        mode_labels = {
            "distributed": i18n.t('distributed'),
            "encrypted":   i18n.t('encrypted'),
        }
        badge = tk.Label(toolbar, text=f"  {mode_labels[self.mode]}  ",
                         bg=mode_colors[self.mode], fg="white",
                         font=("Arial", 12, "bold"), relief=tk.FLAT, padx=8, pady=3)
        badge.pack(side=tk.LEFT)

        self.run_btn = ttk.Button(toolbar, text=i18n.t("run"),
                                  command=self.run_computation, style="Accent.TButton")
        self.run_btn.pack(side=tk.LEFT, padx=(12, 4))
        self.stop_btn = ttk.Button(toolbar, text=i18n.t("stop"),
                                   command=self.stop_computation, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value=i18n.t("ready"))
        ttk.Label(toolbar, textvariable=self.status_var, foreground="gray").pack(
            side=tk.LEFT, padx=12)

        ttk.Button(toolbar, text="中/EN", width=6,
                   command=self._toggle_lang).pack(side=tk.RIGHT)

        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=14, pady=(6, 14))

        left = ttk.Frame(paned, padding=12)
        right = ttk.Frame(paned, padding=12)
        paned.add(left, weight=1)
        paned.add(right, weight=2)

        self._left_notebook = ttk.Notebook(left)
        self._left_notebook.pack(fill=tk.BOTH, expand=True)

        left_tab_data = ttk.Frame(self._left_notebook, padding=6)
        self._left_notebook.add(left_tab_data, text=i18n.t("data_params_tab"))
        left_tab_dist = ttk.Frame(self._left_notebook, padding=6)
        self._left_notebook.add(left_tab_dist, text=i18n.t("dist_enc_tab"))

        self._build_left(left_tab_data)
        self._build_distributed_panel(left_tab_dist)
        self._build_right(right)

    def _build_left(self, parent):
        # ── Data upload ───────────────────────────────────────────────────
        data_frame = ttk.LabelFrame(parent, text=i18n.t("data"), padding=10)
        data_frame.pack(fill=tk.X, pady=(0, 6))

        row_A = ttk.Frame(data_frame)
        row_A.pack(fill=tk.X, pady=2)
        ttk.Label(row_A, text="A matrix:", width=10).pack(side=tk.LEFT)
        ttk.Entry(row_A, textvariable=self.A_path, width=28).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(row_A, text=i18n.t("browse"), width=9,
                   command=lambda: self._browse_file(self.A_path, "A")).pack(side=tk.LEFT)

        row_y = ttk.Frame(data_frame)
        row_y.pack(fill=tk.X, pady=2)
        ttk.Label(row_y, text="y vector:", width=10).pack(side=tk.LEFT)
        ttk.Entry(row_y, textvariable=self.y_path, width=28).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(row_y, text=i18n.t("browse"), width=9,
                   command=lambda: self._browse_file(self.y_path, "y")).pack(side=tk.LEFT)

        self.data_info_var = tk.StringVar(value=i18n.t("no_data"))
        ttk.Label(data_frame, textvariable=self.data_info_var,
                  foreground="gray", font=("Arial", 8)).pack(anchor=tk.W, pady=(4, 0))

        ttk.Button(data_frame, text=i18n.t("load_data"),
                   command=self.load_data).pack(anchor=tk.W, pady=(6, 0))

        # ── ADMM parameters ───────────────────────────────────────────────
        param_frame = ttk.LabelFrame(parent, text=i18n.t("admm_params"), padding=10)
        param_frame.pack(fill=tk.X, pady=(0, 6))

        params = [
            ("ρ (rho):", self.rho_var),
            ("λ (lambda):", self.lamb_var),
            ("Max iter:", self.max_iter_var),
            ("Tolerance:", self.tol_var),
        ]
        for label, var in params:
            row = ttk.Frame(param_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=12).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=16).pack(side=tk.LEFT)

    def _build_distributed_panel(self, parent):
        dist_frame = ttk.LabelFrame(
            parent,
            text=(i18n.t("enc_session") if self.encrypted else i18n.t("dist_session")),
            padding=12,
        )
        dist_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 8))

        info_frame = ttk.Frame(dist_frame)
        info_frame.pack(fill=tk.X, pady=(0, 6))

        local_ip = _get_local_ip()
        ttk.Label(info_frame, text=i18n.t("my_ip"), width=12).pack(side=tk.LEFT)
        self.ip_var = tk.StringVar(value=local_ip)
        ttk.Entry(info_frame, textvariable=self.ip_var, width=16).pack(side=tk.LEFT)
        ttk.Label(info_frame, text=f"  {i18n.t('port')}", width=8).pack(side=tk.LEFT)
        self.reg_port_var = tk.IntVar(value=self.REG_PORT)
        ttk.Entry(info_frame, textvariable=self.reg_port_var, width=7).pack(side=tk.LEFT)

        tok_frame = ttk.Frame(dist_frame)
        tok_frame.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(tok_frame, text=i18n.t("invite_code"), width=12).pack(side=tk.LEFT)
        self.token_var = tk.StringVar(value=self.session_token)
        ttk.Entry(tok_frame, textvariable=self.token_var,
                  width=10, font=("Courier", 13, "bold"), state="readonly").pack(side=tk.LEFT)
        ttk.Button(tok_frame, text=i18n.t("regen_token"),
                   command=self._regen_token).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(dist_frame, text=i18n.t("share_info"),
                  foreground="gray", font=("Arial", 8)).pack(anchor=tk.W, pady=(0, 4))

        # ── Edge node list with per-node column count ──────────────────────
        node_frame = ttk.LabelFrame(dist_frame, text=i18n.t("col_split"), padding=8)
        node_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        ttk.Label(node_frame, text=i18n.t("col_split_hint"),
                  foreground="gray", font=("Arial", 8)).pack(anchor=tk.W, pady=(0, 4))

        node_canvas = tk.Canvas(node_frame, height=120, highlightthickness=0)
        node_canvas.pack(fill=tk.BOTH, expand=True)
        self._edge_inner = ttk.Frame(node_canvas)
        self._edge_inner_id = node_canvas.create_window(
            (0, 0), window=self._edge_inner, anchor="nw")
        node_canvas.bind("<Configure>",
                         lambda e: node_canvas.itemconfig(
                             self._edge_inner_id, width=e.width))
        self._edge_inner.bind("<Configure>",
                               lambda e: node_canvas.configure(
                                   scrollregion=node_canvas.bbox("all")))
        sb = ttk.Scrollbar(node_frame, orient=tk.VERTICAL,
                           command=node_canvas.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        node_canvas.config(yscrollcommand=sb.set)

        # Summary row: master cols remaining
        summary_row = ttk.Frame(node_frame)
        summary_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(summary_row, text=i18n.t("master_cols"), width=14).pack(side=tk.LEFT)
        self.master_cols_var = tk.StringVar(value="—")
        ttk.Label(summary_row, textvariable=self.master_cols_var,
                  font=("Courier", 10, "bold"), width=8).pack(side=tk.LEFT)
        self.split_hint_var = tk.StringVar(value="")
        ttk.Label(summary_row, textvariable=self.split_hint_var,
                  foreground="gray", font=("Arial", 8)).pack(side=tk.LEFT, padx=6)

        self._edge_rows: dict = {}
        for url in self.edge_urls:
            self._add_edge_row(url)
        self._refresh_split_summary()

        btn_row = ttk.Frame(dist_frame)
        btn_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Button(btn_row, text=i18n.t("add_manually"),
                   command=self._add_edge_manually).pack(side=tk.LEFT)

        if self.encrypted:
            enc_frame = ttk.Frame(dist_frame)
            enc_frame.pack(fill=tk.X, pady=(6, 0))
            ttk.Label(enc_frame, text=i18n.t("key_bits"), width=12).pack(side=tk.LEFT)
            self.key_bits_var = tk.IntVar(value=512)
            ttk.Combobox(enc_frame, textvariable=self.key_bits_var,
                         values=[256, 512, 1024, 2048],
                         state="readonly", width=8).pack(side=tk.LEFT)
            ttk.Label(enc_frame, text=i18n.t("key_bits_hint"),
                      foreground="gray", font=("Arial", 8)).pack(side=tk.LEFT, padx=6)

    def _add_edge_row(self, url: str):
        if not hasattr(self, "_edge_inner"):
            return
        if url not in self.col_vars:
            self.col_vars[url] = tk.IntVar(value=0)
        var = self.col_vars[url]
        short = url.replace("http://", "")

        card = tk.Frame(self._edge_inner, bg="#e8f5e9",
                        highlightbackground="#27ae60", highlightthickness=2,
                        relief=tk.FLAT)
        card.pack(fill=tk.X, pady=3, padx=2)

        tk.Label(card, text="●", fg="#27ae60", bg="#e8f5e9",
                 font=("Arial", 10)).pack(side=tk.LEFT, padx=(6, 2))
        tk.Label(card, text=short, bg="#e8f5e9",
                 font=("Courier", 9, "bold"), anchor="w").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        ttk.Label(card, text=i18n.t("cols_label")).pack(side=tk.LEFT)
        col_entry = ttk.Entry(card, textvariable=var, width=6)
        col_entry.pack(side=tk.LEFT, padx=(2, 4))
        var.trace_add("write", lambda *_: self._refresh_split_summary())

        from gui.tooltip import ToolTip
        remove_btn = tk.Button(card, text="✕", fg="#e74c3c", bg="#e8f5e9",
                               activebackground="#fdecea", activeforeground="#c0392b",
                               relief=tk.FLAT, font=("Arial", 9, "bold"), cursor="hand2",
                               command=lambda u=url: self._remove_edge_by_url(u))
        remove_btn.pack(side=tk.RIGHT, padx=(0, 6), pady=4)
        ToolTip(remove_btn, i18n.t("disconnect_tip"))

        self._edge_rows[url] = card

    def _remove_edge_row(self, url: str):
        if url in self._edge_rows:
            self._edge_rows.pop(url).destroy()
        self.col_vars.pop(url, None)
        self._refresh_split_summary()

    def _refresh_split_summary(self):
        if not hasattr(self, "master_cols_var"):
            return
        N = self.A.shape[1] if self.A is not None else None
        if N is None:
            self.master_cols_var.set("—")
            self.split_hint_var.set(i18n.t("load_data_first_hint"))
            return
        total_edge = 0
        valid = True
        for url in self.edge_urls:
            var = self.col_vars.get(url)
            if var is None:
                valid = False
                break
            try:
                v = int(var.get())
                if v < 1:
                    valid = False
                    break
                total_edge += v
            except (tk.TclError, ValueError):
                valid = False
                break
        if not valid:
            self.master_cols_var.set("?")
            self.split_hint_var.set(i18n.t("split_invalid"))
            return
        remaining = N - total_edge
        self.master_cols_var.set(str(remaining) if remaining >= 0 else "!")
        if remaining < 0:
            self.split_hint_var.set(i18n.t("split_overflow").format(N=N))
        elif total_edge == N:
            self.split_hint_var.set(f"✓  {i18n.t('split_ok').format(N=N)}")
        else:
            self.split_hint_var.set(i18n.t("split_invalid"))

    def _auto_split_equal(self):
        """Distribute N columns evenly across all edges when A is loaded."""
        if self.A is None or len(self.edge_urls) == 0:
            return
        N = self.A.shape[1]
        K = len(self.edge_urls)
        base = N // K
        remainder = N % K
        for i, url in enumerate(self.edge_urls):
            var = self.col_vars.get(url)
            if var is not None:
                var.set(base + (1 if i < remainder else 0))
        self._refresh_split_summary()

    def _build_right(self, parent):
        plot_frame = ttk.LabelFrame(parent, text=i18n.t("convergence"), padding=12)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 10))

        self.fig = Figure(figsize=(5, 3), dpi=96)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("||Ax-y||²/M")
        self.ax.set_yscale("log")
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        prog_frame = ttk.Frame(parent)
        prog_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(prog_frame, text=i18n.t("iteration_label")).pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(prog_frame, mode="determinate", maximum=100)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        self.iter_label = ttk.Label(prog_frame, text="0/0", width=10)
        self.iter_label.pack(side=tk.LEFT, padx=(4, 0))

        res_frame = ttk.LabelFrame(parent, text=i18n.t("results"), padding=12)
        res_frame.pack(fill=tk.X, pady=(0, 10))

        self.result_text = tk.Text(res_frame, height=7, state=tk.DISABLED,
                                   bg="#ffffff", fg="#111827", insertbackground="#111827",
                                   font=("Courier", 10), relief=tk.FLAT, highlightthickness=1,
                                   highlightcolor="#e5e7eb", highlightbackground="#e5e7eb")
        self.result_text.pack(fill=tk.X)

        btn_row = ttk.Frame(res_frame)
        btn_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(btn_row, text=i18n.t("save_x"),
                   command=self._save_x).pack(side=tk.LEFT)
        ttk.Button(btn_row, text=i18n.t("save_residuals"),
                   command=self._save_residuals).pack(side=tk.LEFT, padx=6)

        log_frame = ttk.LabelFrame(parent, text=i18n.t("log"), padding=12)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        self.log_text = tk.Text(log_frame, height=10, state=tk.DISABLED,
                                bg="#f8fafc", fg="#111827", insertbackground="#111827",
                                font=("Courier", 9), relief=tk.FLAT, highlightthickness=1,
                                highlightcolor="#e5e7eb", highlightbackground="#e5e7eb")
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb2 = ttk.Scrollbar(log_frame, orient=tk.VERTICAL,
                             command=self.log_text.yview)
        sb2.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=sb2.set)
        self.log_text.tag_config("ERR", foreground="#f44747")
        self.log_text.tag_config("OK",  foreground="#6a9955")


    # ── Language toggle ───────────────────────────────────────────────────────

    def _toggle_lang(self):
        if self._running:
            return
        new_lang = "en" if i18n.get_lang() == "zh" else "zh"
        i18n.set_lang(new_lang)
        cfg.set("lang", new_lang)
        self._refresh_labels()

    def _refresh_labels(self):
        a_path = self.A_path.get()
        y_path = self.y_path.get()
        rho = self.rho_var.get()
        lamb = self.lamb_var.get()
        max_iter = self.max_iter_var.get()
        tol = self.tol_var.get()
        status = self.status_var.get()
        data_info = self.data_info_var.get()
        ip = getattr(self, "ip_var", None)
        ip_val = ip.get() if ip else None
        reg_port = getattr(self, "reg_port_var", None)
        reg_port_val = reg_port.get() if reg_port else None
        key_bits = getattr(self, "key_bits_var", None)
        key_bits_val = key_bits.get() if key_bits else None
        # Preserve per-edge column counts
        saved_cols = {url: var.get() for url, var in self.col_vars.items()}

        for w in self.root.winfo_children():
            w.destroy()

        self._build_ui()
        decode_widget_tree(self.root)

        self.A_path.set(a_path)
        self.y_path.set(y_path)
        self.rho_var.set(rho)
        self.lamb_var.set(lamb)
        self.max_iter_var.set(max_iter)
        self.tol_var.set(tol)
        self.status_var.set(status)
        self.data_info_var.set(data_info)
        if ip_val is not None and hasattr(self, "ip_var"):
            self.ip_var.set(ip_val)
        if reg_port_val is not None and hasattr(self, "reg_port_var"):
            self.reg_port_var.set(reg_port_val)
        if key_bits_val is not None and hasattr(self, "key_bits_var"):
            self.key_bits_var.set(key_bits_val)
        for url, val in saved_cols.items():
            if url in self.col_vars:
                self.col_vars[url].set(val)
        self._refresh_split_summary()

    # ── Event handlers ────────────────────────────────────────────────────────

    def _browse_file(self, var: tk.StringVar, label: str):
        path = filedialog.askopenfilename(
            title=f"Select {label} file",
            filetypes=[("Data files", "*.npy *.mat *.csv *.h5 *.hdf5"),
                       ("All files", "*.*")],
        )
        if path:
            var.set(path)

    def load_data(self):
        a_path = self.A_path.get().strip()
        y_path = self.y_path.get().strip()
        if not a_path or not y_path:
            messagebox.showerror("Error", "Please select both A and y files.")
            return
        try:
            self.log("Loading A...")
            A = _load_array(a_path)
            if A.ndim != 2:
                raise ValueError(f"A must be 2-D, got shape {A.shape}")
            self.log("Loading y...")
            y = _load_array(y_path)
            if y.ndim == 2:
                y = y.flatten()
            if y.ndim != 1:
                raise ValueError(f"y must be 1-D, got shape {y.shape}")
            if A.shape[0] != len(y):
                raise ValueError(
                    f"Dimension mismatch: A has {A.shape[0]} rows but y has {len(y)} elements")
            self.A = A.astype(float)
            self.y = y.astype(float)
            info = f"A: {A.shape[0]}×{A.shape[1]}   y: {len(y)}"
            self.data_info_var.set(info)
            self.log(f"Data loaded — {info}", tag="OK")
            self._auto_split_equal()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            self.log(f"Load failed: {e}", tag="ERR")

    def _regen_token(self):
        self.session_token = _gen_token()
        self.token_var.set(self.session_token)
        if self.reg_server:
            self.reg_server.token = self.session_token
        self.log(f"New invite code: {self.session_token}")

    def _add_edge_manually(self):
        dlg = tk.Toplevel(self.root)
        dlg.title(i18n.t("add_edge_title"))
        dlg.geometry("340x120")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()

        ttk.Label(dlg, text="Edge URL (e.g. http://192.168.1.10:5000):").pack(
            padx=12, pady=(14, 4))
        url_var = tk.StringVar(value="http://")
        ttk.Entry(dlg, textvariable=url_var, width=36).pack(padx=12)

        def _ok():
            url = url_var.get().strip()
            if url and url not in self.edge_urls:
                self.edge_urls.append(url)
                self._add_edge_row(url)
                self._auto_split_equal()
                self.log(f"Edge added: {url}", tag="OK")
            dlg.destroy()

        ttk.Button(dlg, text=i18n.t("add"), command=_ok).pack(pady=10)

    def _remove_edge_by_url(self, url: str):
        if url in self.edge_urls:
            self.edge_urls.remove(url)
        self._remove_edge_row(url)
        self.log(f"Edge removed: {url}")

    # ── Registration server ───────────────────────────────────────────────────

    def _find_free_port(self, start: int, attempts: int = 20) -> int | None:
        import socket as _sock
        for port in range(start, start + attempts):
            with _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM) as s:
                s.setsockopt(_sock.SOL_SOCKET, _sock.SO_REUSEADDR, 1)
                try:
                    s.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    continue
        return None

    def _start_reg_server(self):
        try:
            preferred = int(self.reg_port_var.get())
        except Exception:
            preferred = self.REG_PORT

        port = self._find_free_port(preferred)
        if port is None:
            self.log(
                f"No available port found in range {preferred}–{preferred+19}. "
                f"Change the port and restart.",
                tag="ERR")
            return

        if port != preferred:
            self.log(f"Port {preferred} unavailable, using {port} instead.")
            self.reg_port_var.set(port)

        self.reg_server = MasterRegistrationServer(
            token=self.session_token, port=port)
        self.reg_server.set_on_join(self._on_edge_joined)
        self.reg_server.set_on_token_mismatch(
            lambda exp, got: self.root.after(0, lambda: self.log(
                f"Token mismatch — expected: {exp!r}, received: {got!r}", tag="ERR")))
        try:
            self.reg_server.start()
            self.log(f"Registration server started on port {port}", tag="OK")
        except Exception as e:
            self.log(f"Could not start registration server: {e!r}", tag="ERR")

    def _on_edge_joined(self, edge_url: str):
        if self._alive:
            self.root.after(0, lambda: self._add_edge_url(edge_url))

    def _add_edge_url(self, edge_url: str):
        if edge_url not in self.edge_urls:
            self.edge_urls.append(edge_url)
            self._add_edge_row(edge_url)
            self._auto_split_equal()
            self.log(f"Edge joined: {edge_url}", tag="OK")

    # ── Computation ───────────────────────────────────────────────────────────

    def run_computation(self):
        if self._running:
            return
        if self.A is None or self.y is None:
            messagebox.showerror("Error", i18n.t("load_data_first"))
            return
        if len(self.edge_urls) == 0:
            messagebox.showerror("Error", i18n.t("no_edges_error"))
            return

        # Validate column split: each edge must have >= 1 col, total must equal N
        N = self.A.shape[1]
        edge_cols = []
        for url in self.edge_urls:
            var = self.col_vars.get(url)
            try:
                c = int(var.get()) if var else 0
                if c < 1:
                    raise ValueError()
                edge_cols.append(c)
            except (ValueError, tk.TclError):
                messagebox.showerror("Error",
                    f"Invalid column count for edge {url}.\nEach edge must have ≥ 1 column.")
                return
        if sum(edge_cols) != N:
            messagebox.showerror("Error",
                f"Column counts sum to {sum(edge_cols)} but A has N={N} columns.\n"
                "Adjust the column split so all columns are assigned.")
            return

        try:
            rho = float(self.rho_var.get())
            lamb = float(self.lamb_var.get())
            max_iter = int(self.max_iter_var.get())
            tol = float(self.tol_var.get())
        except Exception as e:
            messagebox.showerror("Parameter Error", str(e))
            return

        self._running = True
        self._stop_flag = False
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.result_x = None
        self.residual_history = []
        self._clear_plot()
        self._clear_results()
        self.progress["maximum"] = max_iter
        self.progress["value"] = 0
        self.iter_label.config(text=f"0/{max_iter}")

        t = threading.Thread(
            target=self._compute_thread,
            args=(rho, lamb, max_iter, tol),
            daemon=True,
        )
        t.start()

    def stop_computation(self):
        self._stop_flag = True
        self.log("Stop requested...")

    def _compute_thread(self, rho, lamb, max_iter, tol):
        try:
            self._run_distributed(rho, lamb, max_iter, tol)
        except Exception as e:
            msg = str(e)
            if self._alive:
                self.root.after(0, lambda m=msg: self.log(f"Error: {m}", tag="ERR"))
                self.root.after(0, lambda m=msg: messagebox.showerror("Computation Error", m))
        finally:
            if self._alive:
                self.root.after(0, self._on_compute_done)

    def _run_distributed(self, rho, lamb, max_iter, tol):
        from core.distributed_admm import FederatedADMMCoordinator
        from network.client import EdgeClient, NetworkError
        from core.quantization import quantize_gamma2
        from utils.gpu import is_gpu_available

        key_bits = getattr(self, "key_bits_var", None)
        key_bits = int(key_bits.get()) if key_bits else 512

        K = len(self.edge_urls)
        A, y = self.A, self.y
        M = len(y)

        # Build column-split A_parts from user-specified col counts
        edge_cols = [int(self.col_vars[url].get()) for url in self.edge_urls]
        col_starts = [sum(edge_cols[:i]) for i in range(K)]
        A_parts = [A[:, col_starts[k]:col_starts[k] + edge_cols[k]] for k in range(K)]

        gpu_msg = "GPU (CuPy) enabled" if is_gpu_available() else "Running on CPU"
        self.root.after(0, lambda: self.log(gpu_msg))
        self.root.after(0, lambda: self.log(
            f"Starting {'encrypted' if self.encrypted else 'distributed'} ADMM "
            f"with {K} edge(s), column split: {edge_cols}..."))

        coordinator = FederatedADMMCoordinator(
            rho=rho, lamb=lamb, max_iter=max_iter, tol=tol,
            encrypted=self.encrypted,
            key_bits=key_bits if self.encrypted else 512,
        )

        clients = [EdgeClient(url, session_token=self.session_token)
                   for url in self.edge_urls]

        # ── Phase 0: push A_k to each edge ────────────────────────────────
        for k, (client, Ak) in enumerate(zip(clients, A_parts)):
            self.root.after(0, lambda i=k: self.log(f"Pushing A_{i} to Edge {i}..."))
            resp = client.send_Ak(Ak.tolist())
            if not resp.get("ok"):
                raise RuntimeError(f"Edge {k} receive_Ak failed: {resp}")

        # ── Phase 1: send y to each edge, collect Ns ──────────────────────
        Ns = []
        edge_quant_params = []

        for k, client in enumerate(clients):
            self.root.after(0, lambda i=k: self.log(f"Initializing Edge {i}..."))
            resp = client.init_local(
                y=y.tolist(), rho=rho,
                encrypted=self.encrypted,
                max_iter=max_iter,
                Delta=coordinator.delta,
                Delta2=coordinator.delta2,
            )
            if not resp.get("ok"):
                raise RuntimeError(f"Edge {k} init_local failed: {resp}")
            Nk = int(resp["Nk"])
            Ns.append(Nk)
            if self.encrypted:
                edge_quant_params.append({
                    "alpha_q":   resp["alpha_q"],
                    "alpha_min": resp["alpha_min"],
                    "alpha_max": resp["alpha_max"],
                    "Bbar_q":    resp["Bbar_q"],
                    "Bbar_min":  resp["Bbar_min"],
                    "Bbar_max":  resp["Bbar_max"],
                })

        # ── Phase 2: setup coordinator with known Ns ──────────────────────
        key_info = coordinator.setup(Ns)

        # ── Phase 3 (encrypted only): encrypt alpha_q, send back ──────────
        if self.encrypted:
            kp = coordinator.keypair
            for k, client in enumerate(clients):
                qp = edge_quant_params[k]
                alpha_hat = [kp.encrypt(int(v)) for v in qp["alpha_q"]]
                coordinator.alpha_params[k] = (qp["alpha_min"], qp["alpha_max"])
                resp = client.init_params_encrypted(
                    alpha_hat=alpha_hat,
                    n=key_info["n"], g=key_info["g"],
                    Delta=key_info["delta"], Delta2=key_info["delta2"],
                    alpha_min=qp["alpha_min"], alpha_max=qp["alpha_max"],
                    Bbar_min=qp["Bbar_min"], Bbar_max=qp["Bbar_max"],
                )
                if not resp.get("ok"):
                    raise RuntimeError(f"Edge {k} init_params failed: {resp}")

        self.root.after(0, lambda: self.log(
            f"Partition sizes: {Ns}  total N={sum(Ns)}"))

        # ── ADMM iteration loop ───────────────────────────────────────────
        residuals = []
        Ax_parts = [np.zeros(M) for _ in range(K)]

        for t in range(1, max_iter + 1):
            if self._stop_flag:
                self.root.after(0, lambda: self.log("Stopped by user."))
                break

            x_new_parts = []

            for k, client in enumerate(clients):
                zk = coordinator.z_parts[k]
                vk = coordinator.v_parts[k]

                if self.encrypted:
                    kp = coordinator.keypair
                    delta = coordinator.delta
                    z_q, _, _ = quantize_gamma2(zk, delta)
                    negv_q, _, _ = quantize_gamma2(-vk, delta)
                    z_hat = [kp.encrypt(int(m)) for m in z_q]
                    v_hat = [kp.encrypt(int(m)) for m in negv_q]
                    resp = client.compute_x(z_hat=z_hat, v_hat=v_hat)
                    if not resp.get("ok"):
                        raise RuntimeError(f"Edge {k} compute_x failed: {resp}")
                    x_hat = resp["x_hat"]
                    a_min, a_max = coordinator.alpha_params[k]
                    x_q = np.array(
                        [v - kp.n if v > kp.n // 2 else v for v in x_hat],
                        dtype=float)
                    x_q = np.clip(x_q / coordinator.delta2, 0, 1)
                    x_k = x_q * (a_max - a_min) + a_min
                    Ax_parts[k] = np.zeros(M)
                else:
                    resp = client.compute_x(z_k=zk.tolist(), v_k=vk.tolist())
                    if not resp.get("ok"):
                        raise RuntimeError(f"Edge {k} compute_x failed: {resp}")
                    x_k = np.array(resp["x_k"])
                    Ax_parts[k] = np.array(resp["Ax_k"])

                x_new_parts.append(x_k)

            coordinator.z_v_update(x_new_parts)

            if not self.encrypted:
                res = float(np.linalg.norm(sum(Ax_parts) - y) ** 2 / M)
            else:
                x_full = np.concatenate(x_new_parts)
                z_full = np.concatenate(coordinator.z_parts)
                res = float(np.linalg.norm(x_full - z_full) ** 2 / len(x_full))
            residuals.append(res)

            if t % 5 == 0 or t == max_iter:
                snap = residuals[:]
                self.root.after(0, lambda it=t, r=snap: self._update_progress(
                    it, max_iter, r))

            if t > 1 and abs(residuals[-1] - residuals[-2]) / (residuals[-2] + 1e-12) < tol:
                self.root.after(0, lambda it=t: self.log(
                    f"Converged at iteration {it}", tag="OK"))
                break
        else:
            self.root.after(0, lambda: self.log(
                f"Reached max_iter={max_iter} without convergence. "
                f"Consider increasing Max iter or relaxing Tolerance.", tag="ERR"))

        x_final = np.concatenate(coordinator.x_parts)
        self.result_x = x_final
        self.residual_history = residuals
        self.root.after(0, lambda: self._show_results(x_final, residuals))


    # ── UI update helpers ─────────────────────────────────────────────────────

    def _update_progress(self, iteration: int, max_iter: int, residuals: list):
        self.progress["maximum"] = max_iter
        self.progress["value"] = iteration
        self.iter_label.config(text=f"{iteration}/{max_iter}")
        self.status_var.set(_decode_unicode(
            f"Iter {iteration}/{max_iter}  Residual={residuals[-1]:.3e}"))
        self._redraw_plot(residuals)

    def _redraw_plot(self, residuals: list):
        self.ax.clear()
        self.ax.plot(range(1, len(residuals) + 1), residuals, "b-", linewidth=1.2)
        self.ax.set_xlabel("Iteration")
        ylabel = "||Ax-y||²/M" if not self.encrypted else "||x-z||²/N"
        self.ax.set_ylabel(ylabel)
        self.ax.set_yscale("log")
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _clear_plot(self):
        self.ax.clear()
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Residual")
        self.ax.set_yscale("log")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def _clear_results(self):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.config(state=tk.DISABLED)

    def _show_results(self, x: np.ndarray, residuals: list):
        self._redraw_plot(residuals)
        max_iter = self.max_iter_var.get()
        self.progress["maximum"] = max_iter
        self.progress["value"] = len(residuals)
        self.iter_label.config(text=f"{len(residuals)}/{max_iter}")

        lines = [
            f"Iterations   : {len(residuals)}",
            f"Final residual: {residuals[-1]:.6e}",
            f"x shape      : {x.shape}",
            f"||x||_0 (nnz): {int(np.sum(np.abs(x) > 1e-6))}",
            f"||x||_1      : {float(np.sum(np.abs(x))):.4f}",
        ]
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "\n".join(lines))
        self.result_text.config(state=tk.DISABLED)
        self.log("Computation complete.", tag="OK")

    def _on_compute_done(self):
        self._running = False
        self.run_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set(i18n.t("done") if not self._stop_flag else i18n.t("stopped"))

    # ── Export ────────────────────────────────────────────────────────────────

    def _save_x(self):
        if self.result_x is None:
            messagebox.showinfo("Info", "No result to save yet.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".npy",
            filetypes=[("NumPy array", "*.npy"), ("All files", "*.*")],
            initialfile="x_result.npy",
        )
        if path:
            np.save(path, self.result_x)
            self.log(f"Saved x → {path}", tag="OK")

    def _save_residuals(self):
        if not self.residual_history:
            messagebox.showinfo("Info", "No residuals to save yet.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            initialfile="residuals.csv",
        )
        if path:
            arr = np.array(self.residual_history)
            np.savetxt(path, arr.reshape(-1, 1), delimiter=",",
                       header="residual", comments="")
            self.log(f"Saved residuals → {path}", tag="OK")

    # ── Logging ───────────────────────────────────────────────────────────────

    def log(self, msg: str, tag: str = ""):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {_decode_unicode(msg)}\n"
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, line, tag)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        logger.info(msg)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_close(self):
        self._alive = False
        self._stop_flag = True
        if self.reg_server:
            self.reg_server.stop()
        self.root.destroy()

