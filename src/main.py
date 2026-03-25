#!/usr/bin/env python
"""
3P-ADMM-PC2 — Launcher

Double-click this file to start. Choose your mode in the startup window.
For Distributed/Encrypted modes, you will also choose your role (Master / Edge).
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
import threading
import subprocess

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from gui.style import setup_style, decode_widget_tree
from gui import config as cfg
from gui import i18n


def _init_lang():
    """Load persisted language or show first-run dialog."""
    cfg.load()
    saved = cfg.get("lang")
    if saved in ("zh", "en"):
        i18n.set_lang(saved)
        return
    tmp_root = tk.Tk()
    tmp_root.withdraw()
    _show_lang_dialog(tmp_root)
    tmp_root.destroy()


def _show_lang_dialog(parent: tk.Tk):
    """Modal language selection dialog. Blocks until user picks."""
    dlg = tk.Toplevel(parent)
    dlg.title("3P-ADMM-PC2")
    dlg.geometry("340x200")
    dlg.resizable(False, False)
    dlg.grab_set()

    dlg.update_idletasks()
    x = (dlg.winfo_screenwidth() - 340) // 2
    y = (dlg.winfo_screenheight() - 200) // 2
    dlg.geometry(f"340x200+{x}+{y}")

    tk.Label(dlg, text=i18n.STRINGS["zh"]["choose_lang"],
             font=("Arial", 12), justify=tk.CENTER).pack(pady=(28, 20))

    btn_frame = ttk.Frame(dlg)
    btn_frame.pack()

    chosen = [None]

    def _pick(lang):
        chosen[0] = lang
        dlg.destroy()

    ttk.Button(btn_frame, text="中文", width=12,
               command=lambda: _pick("zh")).pack(side=tk.LEFT, padx=12)
    ttk.Button(btn_frame, text="English", width=12,
               command=lambda: _pick("en")).pack(side=tk.LEFT, padx=12)

    dlg.protocol("WM_DELETE_WINDOW", lambda: _pick("zh"))
    parent.wait_window(dlg)

    lang = chosen[0] or "zh"
    i18n.set_lang(lang)
    cfg.set("lang", lang)


# ── CuPy auto-detect & install prompt ────────────────────────────────────────

def _check_gpu_and_prompt(root: tk.Tk) -> None:
    """
    Background check: if an NVIDIA GPU is present but CuPy is not working,
    show a dialog explaining how to enable GPU acceleration.
    """
    if cfg.get("skip_cupy_prompt"):
        return

    def _worker():
        # 1. Is there an NVIDIA GPU?
        gpu_name = None
        cuda_ver = None
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            raw = pynvml.nvmlDeviceGetName(handle)
            gpu_name = raw.decode() if isinstance(raw, bytes) else raw
            v = pynvml.nvmlSystemGetCudaDriverVersion()
            cuda_ver = (v // 1000, (v % 1000) // 10)
        except Exception:
            return  # No NVIDIA GPU — nothing to do

        # 2. Try to load CuPy (including from system Python)
        from utils.gpu import is_gpu_available, get_gpu_error
        if is_gpu_available():
            return  # CuPy is working

        # 3. CuPy not working but GPU exists — show dialog
        error_msg = get_gpu_error()
        root.after(0, lambda: _show_cupy_dialog(root, gpu_name, cuda_ver, error_msg))

    threading.Thread(target=_worker, daemon=True).start()


def _show_cupy_dialog(root: tk.Tk, gpu_name: str, cuda_ver, error_msg: str = "") -> None:
    """Dialog that explains CuPy and offers to install it."""
    zh = i18n.get_lang() == "zh"
    is_frozen = getattr(sys, 'frozen', False)

    # Choose the right pip package based on CUDA major version
    pkg = None
    cuda_str = ""
    if cuda_ver:
        major, minor = cuda_ver
        cuda_str = f"CUDA {major}.{minor}"
        if major >= 12:
            pkg = "cupy-cuda12x"
        elif major == 11:
            pkg = "cupy-cuda11x"

    dlg = tk.Toplevel(root)
    dlg.title("GPU 加速 / GPU Acceleration")
    dlg.geometry("520x480")
    dlg.resizable(False, False)
    dlg.transient(root)

    dlg.update_idletasks()
    x = root.winfo_x() + (root.winfo_width()  - 520) // 2
    y = root.winfo_y() + (root.winfo_height() - 430) // 2
    dlg.geometry(f"520x430+{x}+{y}")

    # ── Header ────────────────────────────────────────────────────────────
    hdr = ttk.Frame(dlg, padding=(20, 18, 20, 6))
    hdr.pack(fill=tk.X)

    icon_text = "🎮  " if not zh else "🎮  "
    title_text = ("检测到 NVIDIA GPU" if zh else "NVIDIA GPU Detected")
    ttk.Label(hdr, text=title_text,
              font=("Arial", 14, "bold")).pack(anchor=tk.W)

    gpu_line = gpu_name + (f"  ·  {cuda_str}" if cuda_str else "")
    ttk.Label(hdr, text=gpu_line, foreground="gray",
              font=("Arial", 9)).pack(anchor=tk.W, pady=(2, 0))

    ttk.Separator(dlg).pack(fill=tk.X, padx=20, pady=(8, 0))

    # ── Body ──────────────────────────────────────────────────────────────
    body = ttk.Frame(dlg, padding=(20, 12, 20, 4))
    body.pack(fill=tk.X)

    if zh:
        if is_frozen:
            lines = (
                "CuPy 是 NumPy 的 GPU 加速替代品，安装后程序会自动检测并使用。\n\n"
                "用途与效果：\n"
                "  • 矩阵求逆、矩阵乘法等运算：通常快 5–20 倍\n"
                "  • 适用于分布式模式的本地计算\n\n"
                "安装方法：在系统 Python 中安装 CuPy，重启程序即可。\n"
                f"命令：pip install {pkg or 'cupy-cuda12x'}"
            )
        else:
            lines = (
                "CuPy 是 NumPy 的 GPU 加速替代品，安装后无需改动代码即可自动启用。\n\n"
                "用途与效果：\n"
                "  • 矩阵求逆、矩阵乘法等运算：通常快 5–20 倍\n"
                "  • 适用于分布式模式的本地计算（Master 和 Edge 均受益）\n"
                "  • 加密模式中 Paillier 运算仍在 CPU，GPU 加速效果有限\n\n"
                "安装完成后重启程序即可自动使用 GPU。"
            )
        if pkg is None:
            lines += "\n\n⚠ 未能自动识别 CUDA 版本，请手动选择安装命令。"
    else:
        if is_frozen:
            lines = (
                "CuPy is a GPU-accelerated drop-in replacement for NumPy.\n"
                "The app will automatically detect and use it.\n\n"
                "What it helps with:\n"
                "  • Matrix inversion, multiplication, solve: typically 5–20× faster\n"
                "  • Benefits both Master and Edge in distributed mode\n\n"
                "Installation: Install CuPy in your system Python, then restart.\n"
                f"Command: pip install {pkg or 'cupy-cuda12x'}"
            )
        else:
            lines = (
                "CuPy is a GPU-accelerated drop-in replacement for NumPy.\n"
                "No code changes needed — the app detects and uses it automatically.\n\n"
                "What it helps with:\n"
                "  • Matrix inversion, multiplication, solve: typically 5–20× faster\n"
                "  • Benefits both Master and Edge in distributed mode\n"
                "  • Encrypted mode (Paillier) still runs on CPU — limited GPU benefit\n\n"
                "Restart the app after installation to activate GPU acceleration."
            )
        if pkg is None:
            lines += "\n\n⚠ Could not auto-detect CUDA version. Choose the command manually."

    ttk.Label(body, text=lines, justify=tk.LEFT,
              wraplength=478).pack(anchor=tk.W)

    # ── Output area (pip log) ─────────────────────────────────────────────
    out_frame = ttk.Frame(dlg, padding=(20, 4, 20, 0))
    out_frame.pack(fill=tk.BOTH, expand=True)

    out_text = tk.Text(
        out_frame, height=5, state=tk.DISABLED,
        bg="#0b1220", fg="#e5e7eb",
        font=("Courier", 8), relief=tk.FLAT,
        highlightthickness=1, highlightbackground="#233044",
    )
    out_text.pack(fill=tk.BOTH, expand=True)
    out_text.tag_config("ok",  foreground="#6a9955")
    out_text.tag_config("err", foreground="#f44747")

    def _append(text, tag=""):
        out_text.config(state=tk.NORMAL)
        out_text.insert(tk.END, text, tag)
        out_text.see(tk.END)
        out_text.config(state=tk.DISABLED)

    # ── Buttons ───────────────────────────────────────────────────────────
    btn_frame = ttk.Frame(dlg, padding=(20, 10, 20, 14))
    btn_frame.pack(fill=tk.X, side=tk.BOTTOM)

    no_ask_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(
        btn_frame,
        text=("不再提示" if zh else "Don't ask again"),
        variable=no_ask_var,
    ).pack(side=tk.LEFT)

    def _on_skip():
        if no_ask_var.get():
            cfg.set("skip_cupy_prompt", True)
        dlg.destroy()

    def _copy_command():
        """Copy the pip install command to clipboard."""
        cmd = f"pip install {pkg or 'cupy-cuda12x'}"
        dlg.clipboard_clear()
        dlg.clipboard_append(cmd)
        msg = ("已复制到剪贴板" if zh else "Copied to clipboard")
        _append(f"$ {cmd}\n{msg}\n", "ok")

    def _on_install():
        if is_frozen:
            # In packaged exe, just copy the command
            _copy_command()
            return

        if pkg is None:
            # No CUDA version detected — show manual commands
            _append("# CUDA 12.x:\npip install cupy-cuda12x\n\n"
                    "# CUDA 11.x:\npip install cupy-cuda11x\n")
            return

        install_btn.config(state=tk.DISABLED)
        skip_btn.config(state=tk.DISABLED)

        def _run_pip():
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "pip", "install", pkg],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                for line in proc.stdout:
                    dlg.after(0, lambda l=line.rstrip(): _append(l + "\n"))
                proc.wait()
                if proc.returncode == 0:
                    msg = ("✓ 安装成功！重启程序以启用 GPU 加速。\n" if zh
                           else "✓ Installed successfully! Restart to enable GPU.\n")
                    dlg.after(0, lambda: _append("\n" + msg, "ok"))
                else:
                    msg = ("✗ 安装失败，请查看上方错误。\n" if zh
                           else "✗ Installation failed. See output above.\n")
                    dlg.after(0, lambda: _append("\n" + msg, "err"))
            except Exception as e:
                dlg.after(0, lambda: _append(f"Error: {e}\n", "err"))
            dlg.after(0, lambda: skip_btn.config(state=tk.NORMAL))

        threading.Thread(target=_run_pip, daemon=True).start()

    # Button labels
    if is_frozen:
        install_label = ("复制安装命令" if zh else "Copy Command")
    elif pkg:
        install_label = (f"安装 {pkg}" if zh else f"Install {pkg}")
    else:
        install_label = ("查看安装命令" if zh else "Show Install Commands")

    skip_btn = ttk.Button(btn_frame,
                          text=("跳过" if zh else "Skip"),
                          command=_on_skip)
    skip_btn.pack(side=tk.RIGHT, padx=(8, 0))

    install_btn = ttk.Button(btn_frame, text=install_label,
                             command=_on_install)
    install_btn.pack(side=tk.RIGHT)

    decode_widget_tree(dlg)
    dlg.protocol("WM_DELETE_WINDOW", _on_skip)


# ── Role dialog ───────────────────────────────────────────────────────────────

class RoleDialog(tk.Toplevel):
    """Ask the user to choose Master or Edge role."""

    def __init__(self, parent, mode: str):
        super().__init__(parent)
        self.result = None
        self.mode = mode

        label = (i18n.t('distributed') if mode == "distributed"
                 else i18n.t('encrypted'))
        self.title(f"{label} — {i18n.t('select_role')}")
        self.geometry("420x220")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 420) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 220) // 2
        self.geometry(f"420x220+{x}+{y}")

        ttk.Label(self, text=i18n.t("select_role"),
                  font=("Arial", 12, "bold")).pack(pady=(24, 16))

        btn_frame = ttk.Frame(self)
        btn_frame.pack()

        f_master = ttk.LabelFrame(btn_frame, text="Master", padding=12)
        f_master.pack(side=tk.LEFT, padx=14)
        ttk.Label(f_master, text=i18n.t("role_master_desc"),
                  foreground="gray", justify=tk.CENTER).pack()
        ttk.Button(f_master, text=i18n.t("i_am_master"),
                   command=lambda: self._choose("master")).pack(pady=(10, 0))

        f_edge = ttk.LabelFrame(btn_frame, text="Edge", padding=12)
        f_edge.pack(side=tk.LEFT, padx=14)
        ttk.Label(f_edge, text=i18n.t("role_edge_desc"),
                  foreground="gray", justify=tk.CENTER).pack()
        ttk.Button(f_edge, text=i18n.t("i_am_edge"),
                   command=lambda: self._choose("edge")).pack(pady=(10, 0))

        decode_widget_tree(self)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _choose(self, role: str):
        self.result = role
        self.destroy()


# ── Launcher window ───────────────────────────────────────────────────────────

class LauncherWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("3P-ADMM-PC2")
        self.root.geometry("640x520")
        self.root.resizable(False, False)
        self._center()
        setup_style(self.root)
        self._build_ui()
        decode_widget_tree(self.root)
        # Check for GPU after the window is fully rendered
        self.root.after(800, lambda: _check_gpu_and_prompt(self.root))

    def _center(self):
        self.root.update_idletasks()
        w, h = 640, 520
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _build_ui(self):
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill=tk.X, padx=48, pady=(28, 0))
        ttk.Label(title_frame, text="3P-ADMM-PC2",
                  font=("Arial", 24, "bold")).pack(anchor=tk.W)
        ttk.Label(
            title_frame,
            text=i18n.t("app_subtitle"),
            font=("Arial", 10), foreground="gray"
        ).pack(anchor=tk.W, pady=(4, 0))

        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=48, pady=16)

        modes_frame = ttk.Frame(self.root)
        modes_frame.pack(fill=tk.X, padx=56)

        f2 = ttk.LabelFrame(modes_frame, text=i18n.t('distributed'), padding=14)
        f2.pack(fill=tk.X, pady=8)
        ttk.Label(f2, text=i18n.t("distributed_desc"),
                  foreground="gray").pack(anchor=tk.W, pady=(0, 8))
        ttk.Button(f2, text=i18n.t("launch"), width=20,
                   command=lambda: self._launch("distributed")).pack(anchor=tk.W)

        f3 = ttk.LabelFrame(modes_frame, text=i18n.t('encrypted'), padding=14)
        f3.pack(fill=tk.X, pady=8)
        ttk.Label(f3, text=i18n.t("encrypted_desc"),
                  foreground="gray").pack(anchor=tk.W, pady=(0, 8))
        ttk.Button(f3, text=i18n.t("launch"), width=20,
                   command=lambda: self._launch("encrypted")).pack(anchor=tk.W)

        ttk.Label(
            self.root,
            text=i18n.t("tip_distributed"),
            font=("Arial", 9), foreground="gray"
        ).pack(side=tk.BOTTOM, pady=16)

    def _launch(self, mode: str):
        dlg = RoleDialog(self.root, mode)
        self.root.wait_window(dlg)
        if dlg.result == "master":
            self._open_master(mode)
        elif dlg.result == "edge":
            self._open_edge(mode)

    def _open_master(self, mode: str):
        try:
            from gui.master_gui import MasterWindow
            self.root.withdraw()
            win = tk.Toplevel(self.root)
            app = MasterWindow(win, mode=mode)
            win.protocol("WM_DELETE_WINDOW",
                         lambda: (app.on_close(), self.root.deiconify()))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open Master window:\n{e}")
            self.root.deiconify()

    def _open_edge(self, mode: str):
        try:
            from gui.edge_gui import EdgeWindow
            self.root.withdraw()
            win = tk.Toplevel(self.root)
            app = EdgeWindow(win, encrypted=(mode == "encrypted"))
            win.protocol("WM_DELETE_WINDOW",
                         lambda: (app.on_close(), self.root.deiconify()))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open Edge window:\n{e}")
            self.root.deiconify()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", lambda: sys.exit(0))
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    _init_lang()
    LauncherWindow().run()
