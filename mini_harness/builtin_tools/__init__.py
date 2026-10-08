"""内建工具：保留 shell/read_file/write_file 兼容入口，扩展工具由 files 与 jobs 插件装配。"""

from .fs import plugin as fs_plugin
from .shell import (
    detect_shell,
    is_windows_app_alias,
    make_shell_tool,
    plugin as shell_plugin,
)

__all__ = [
    "make_shell_tool",
    "shell_plugin",
    "detect_shell",
    "is_windows_app_alias",
    "fs_plugin",
]
