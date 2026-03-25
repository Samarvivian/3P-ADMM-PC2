"""
工具提示（Tooltip）支持

为GUI组件添加工具提示，提升用户体验
"""

import tkinter as tk
from typing import Optional


class ToolTip:
    """
    工具提示类

    为Tkinter组件添加鼠标悬停提示功能
    """

    def __init__(self, widget: tk.Widget, text: str, delay: int = 500):
        """
        初始化工具提示

        Args:
            widget: 要添加提示的组件
            text: 提示文本
            delay: 显示延迟（毫秒）
        """
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip_window: Optional[tk.Toplevel] = None
        self.schedule_id: Optional[str] = None

        # 绑定事件
        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)
        self.widget.bind("<Button>", self.on_leave)

    def on_enter(self, event=None):
        """鼠标进入时调度显示"""
        self.schedule()

    def on_leave(self, event=None):
        """鼠标离开时取消显示"""
        self.unschedule()
        self.hide()

    def schedule(self):
        """调度显示工具提示"""
        self.unschedule()
        self.schedule_id = self.widget.after(self.delay, self.show)

    def unschedule(self):
        """取消调度"""
        if self.schedule_id:
            self.widget.after_cancel(self.schedule_id)
            self.schedule_id = None

    def show(self):
        """显示工具提示"""
        if self.tooltip_window:
            return

        # 获取组件位置
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        # 创建提示窗口
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        # 创建标签
        label = tk.Label(
            self.tooltip_window,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Arial", 9),
            padx=5,
            pady=3,
        )
        label.pack()

    def hide(self):
        """隐藏工具提示"""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


def create_tooltip(widget: tk.Widget, text: str, delay: int = 500) -> ToolTip:
    """
    创建工具提示的便捷函数

    Args:
        widget: 要添加提示的组件
        text: 提示文本
        delay: 显示延迟（毫秒）

    Returns:
        ToolTip对象

    Example:
        >>> button = tk.Button(root, text="Click me")
        >>> create_tooltip(button, "This button does something")
    """
    return ToolTip(widget, text, delay)
