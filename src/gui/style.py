r"""Unified ttk style and typography for 3P-ADMM-PC2 GUIs."""

import sys
import tkinter as tk
from tkinter import ttk, font


def _fix_tcl_encoding(root: tk.Misc) -> None:
    """Force Tcl/Tk to use UTF-8 encoding.

    On Windows with a non-UTF-8 system locale (e.g. cp936/GBK in China),
    Tcl's default system encoding may not be utf-8.  When that happens,
    Chinese characters set via widget text= are round-tripped through the
    wrong codec and come back as \\uXXXX escape sequences.  Setting the
    encoding to utf-8 before any widgets are created prevents this.
    """
    try:
        current = root.tk.call("encoding", "system")
        if current.lower() not in ("utf-8", "utf8"):
            root.tk.call("encoding", "system", "utf-8")
    except Exception:
        pass


def decode_widget_tree(widget: tk.Misc) -> None:
    r"""Walk widget tree and fix any \\uXXXX escape sequences in widget text.

    This is a safety net for environments where Tcl returns escape sequences
    instead of proper Unicode characters.
    """
    if hasattr(widget, "keys"):
        try:
            if "text" in widget.keys():
                t = widget.cget("text")
                if isinstance(t, str) and "\\u" in t:
                    try:
                        widget.configure(text=t.encode("utf-8").decode("unicode_escape"))
                    except Exception:
                        pass
        except Exception:
            pass

    # Notebook tab texts
    if isinstance(widget, ttk.Notebook):
        for tab_id in widget.tabs():
            try:
                t = widget.tab(tab_id, "text")
                if isinstance(t, str) and "\\u" in t:
                    widget.tab(tab_id, text=t.encode("utf-8").decode("unicode_escape"))
            except Exception:
                continue

    for child in getattr(widget, "winfo_children", lambda: [])():
        decode_widget_tree(child)


def setup_style(root: tk.Misc) -> None:
    r"""Apply modern light palette, larger fonts, and widget padding.

    Must be called BEFORE creating any widgets so that the Tcl encoding
    is set correctly from the start.
    """
    # Fix Tcl encoding FIRST — this prevents Chinese text from being
    # stored as \\uXXXX escape sequences on Windows with non-UTF-8 locale.
    _fix_tcl_encoding(root)

    # Slightly enlarge default scaling for readability on HiDPI screens
    try:
        root.tk.call("tk", "scaling", 1.15)
    except Exception:
        pass

    # Preferred font family fallback list (first available wins).
    # Includes both "Microsoft YaHei" and "Microsoft YaHei UI" because
    # different Windows versions ship one or the other.
    families = (
        "Microsoft YaHei UI",   # Windows 10/11 Chinese
        "Microsoft YaHei",      # older Windows Chinese
        "Noto Sans SC",         # Linux / some Windows installs
        "Noto Sans CJK SC",     # Linux CJK meta-package
        "PingFang SC",          # macOS
        "SimHei",               # Windows fallback
        "Arial Unicode MS",     # broad Unicode coverage
        "Arial",                # last resort (no CJK glyphs — avoid if possible)
    )

    def _pick_family():
        available = set(font.families())
        for fam in families:
            if fam in available:
                return fam
        return "Helvetica"

    base_family = _pick_family()

    # Update Tk named fonts
    font.nametofont("TkDefaultFont").configure(family=base_family, size=11)
    font.nametofont("TkTextFont").configure(family=base_family, size=11)
    font.nametofont("TkHeadingFont").configure(family=base_family, size=13, weight="bold")

    # Palette (light, clean)
    colors = {
        "bg": "#f5f7fb",
        "panel": "#ffffff",
        "border": "#e5e7eb",
        "text": "#111827",
        "muted": "#6b7280",
        "accent": "#3b82f6",
        "success": "#16a34a",
        "danger": "#dc2626",
    }

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    style.configure(
        ".",
        background=colors["bg"],
        foreground=colors["text"],
        fieldbackground=colors["panel"],
        bordercolor=colors["border"],
        lightcolor=colors["border"],
        darkcolor=colors["border"],
        troughcolor=colors["panel"],
    )

    style.configure(
        "TLabelFrame",
        background=colors["panel"],
        foreground=colors["text"],
        relief="solid",
        borderwidth=1,
        bordercolor=colors["border"],
        padding=10,
    )
    style.configure("TFrame", background=colors["bg"])
    style.configure("TLabel", background=colors["bg"], foreground=colors["text"])

    style.configure(
        "TButton",
        background=colors["panel"],
        foreground=colors["text"],
        padding=(12, 7),
        borderwidth=1,
        focusthickness=1,
        )
    style.map(
        "TButton",
        background=[("active", colors["accent"]), ("pressed", colors["accent"])],
        foreground=[("active", "white"), ("pressed", "white")],
    )

    style.configure(
        "Accent.TButton",
        background=colors["accent"],
        foreground="white",
        padding=(13, 8),
        borderwidth=0,
    )
    style.map(
        "Accent.TButton",
        background=[("active", "#3f6fdd"), ("pressed", "#3a64c9")],
    )

    style.configure(
        "TEntry",
        fieldbackground=colors["panel"],
        insertcolor=colors["text"],
    )

    style.configure(
        "TCombobox",
        fieldbackground=colors["panel"],
        insertcolor=colors["text"],
        arrowcolor=colors["text"],
    )

    style.configure(
        "TProgressbar",
        troughcolor="#eef2ff",
        background=colors["accent"],
        thickness=12,
        bordercolor=colors["border"],
    )

    # Treeview/Listbox colors won't follow ttk style; they keep defaults.
