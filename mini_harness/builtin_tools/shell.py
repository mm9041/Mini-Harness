"""``shell`` 工具 —— 迷你版唯一一个面向模型的能力。

对应 dsh 的 ``packages/shell/tool-bash`` / ``tool-pwsh``。要点在于:工具自己不
知道怎么执行命令,它只是 ``ctx.shell`` 这个执行接缝的消费者。dsh 里正因为如此,
把执行 provider 换成远端沙箱,Bash / PTY / LSP 会一起搬过去,不需要 fork 任何工具。

迷你版把执行接缝直接内联在工具里(平台自适应),但保留了同样的外部形状:
名字 + 描述 + JSON Schema + 处理器。
"""

from __future__ import annotations

import asyncio
import ntpath
import os
import shutil
import subprocess
from pathlib import Path

from ..kernel import Context, Plugin
from ..tools import Tool, ToolCallContext, ToolResult

__all__ = ["SHELL_PARAMETERS", "detect_shell", "make_shell_tool", "plugin"]

SHELL_PARAMETERS = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "要执行的命令。它是完整的一行,可以直接使用管道与重定向。",
        }
    },
    "required": ["command"],
    "additionalProperties": False,
}

DESCRIPTION = (
    "在本地 shell 中执行一条命令并返回合并后的 stdout/stderr 与退出码。"
    "状态在多次调用之间不保留:每次调用都是新进程。"
)


def _find_on_path(name: str) -> str | None:
    return shutil.which(name)


def _file_exists(path: str) -> bool:
    return os.path.isfile(path)


def _file_size(path: str) -> int:
    """取文件大小;取不到返回 -1(不存在、无权限、reparse point 异常都算)。"""
    try:
        return os.path.getsize(path)
    except OSError:
        return -1


def is_windows_app_alias(path: str) -> bool:
    """判断路径是否是 Windows 的"应用执行别名"。

    ``%LOCALAPPDATA%\\Microsoft\\WindowsApps`` 下的条目是 0 字节的 reparse point。
    对 ``bash`` 而言,这个别名指向 **WSL** —— 选中它意味着所有命令都跑进 Linux 子系统:
    ``uname -s`` 是 ``Linux``,工作目录变成 ``/mnt/e/...``,PATH 与 python 全是 WSL 的。
    在自己的 Windows 工作区里这就是"环境静默错位",所以默认要避开它
    (用户用 ``--shell`` 显式指定时仍会尊重)。

    用 ``ntpath`` 而不是 ``os.path`` 解析,是为了让这个判断在非 Windows 上也能被测试。
    """
    if _file_size(path) != 0:
        return False
    parts = [part.lower() for part in ntpath.normpath(path).split("\\")]
    return len(parts) >= 3 and parts[-2] == "windowsapps" and parts[-3] == "microsoft"


def _bash_shipped_with_git() -> str | None:
    """从 PATH 上的 ``git`` 反推同一套 Git for Windows 里的 bash。

    比硬编码路径稳:Git 装在 D 盘、自定义目录、Scoop / winget 装的都能顺着
    ``...\\Git\\cmd\\git.exe`` 找到 Git 根目录。

    **优先 ``bin\\bash.exe`` 而不是 ``usr\\bin\\bash.exe``** —— 实测(受限 PATH 下):
    ``bin\\bash.exe`` 会把 ``/mingw64/bin`` 与 ``/usr/bin`` 前置进子进程 PATH,
    ``grep``/``sed``/``wc``/``uname`` 都在;而 ``usr\\bin\\bash.exe`` 原样继承 PATH,
    这些 coreutils 全部 ``command not found``,连 ``find`` 都会落到 Windows 自带的
    ``C:\\Windows\\System32\\find.exe``(一个完全不同的程序)。见 ``_coreutils_dirs``。
    """
    git = _find_on_path("git")
    if not git:
        return None
    root = os.path.dirname(os.path.dirname(os.path.realpath(git)))
    for tail in (("bin", "bash.exe"), ("usr", "bin", "bash.exe")):
        candidate = os.path.join(root, *tail)
        if _file_exists(candidate):
            return candidate
    return None


def _well_known_bash_paths() -> list[str]:
    candidates = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\usr\bin\bash.exe",
    ]
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidates.append(os.path.join(local_appdata, "Programs", "Git", "bin", "bash.exe"))
        candidates.append(
            os.path.join(local_appdata, "Programs", "Git", "usr", "bin", "bash.exe")
        )
    return candidates


def _coreutils_dirs(executable: str) -> list[str]:
    """推出这个 bash 自带的 coreutils 目录(Git for Windows 的布局)。

    ``<root>\\bin\\bash.exe`` 与 ``<root>\\usr\\bin\\bash.exe`` 两种入口都要能还原出
    ``<root>``,再给出 ``usr\\bin`` 与 ``mingw64\\bin``。
    """
    location = Path(executable).parent
    if location.name.lower() == "bin" and location.parent.name.lower() == "usr":
        root = location.parent.parent  # ...\Git
    else:
        root = location.parent  # ...\Git
    return [str(root / "usr" / "bin"), str(root / "mingw64" / "bin")]


def _child_environment(label: str, executable: str) -> dict[str, str] | None:
    """给 bash 子进程补上 coreutils 目录;没有可补的就返回 None(原样继承)。

    为什么必须补:模型写的 ``grep`` / ``sed`` / ``wc`` / ``uname`` 都是 coreutils,
    它们不在 Windows 的 PATH 上。补进去(前置,优先于 ``C:\\Windows\\System32``)
    才能保证"选了 bash 就真的有一套 bash 工具",而不是让 ``find`` 悄悄变成
    Windows 的 ``find.exe``。
    """
    if label != "bash":
        return None

    current = os.environ.get("PATH", "")
    present = {
        os.path.normcase(part) for part in current.split(os.pathsep) if part
    }
    extra = [
        directory
        for directory in _coreutils_dirs(executable)
        if os.path.isdir(directory) and os.path.normcase(directory) not in present
    ]
    if not extra:
        return None

    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([*extra, current]) if current else os.pathsep.join(extra)
    return env


def _shell_argv_for(prefer: str) -> tuple[str, list[str]]:
    """把用户显式给的 shell 路径解析成 (标签, argv 前缀)。"""
    name = Path(prefer).name.lower()
    if "pwsh" in name:
        return "pwsh", [prefer, "-NoProfile", "-NonInteractive", "-Command"]
    if "powershell" in name:
        return "powershell", [prefer, "-NoProfile", "-NonInteractive", "-Command"]
    if name.startswith("cmd"):
        return "cmd", [prefer, "/c"]
    return "bash", [prefer, "-c"]


def _resolve_windows_shell() -> tuple[str, list[str]] | None:
    """Windows 上的选择顺序。

    1. PATH 上的 ``bash`` —— 但**排除 WSL 别名**(见 ``is_windows_app_alias``);
    2. 顺着 PATH 上的 ``git`` 找它自带的 Git Bash;
    3. 常见安装位置;
    4. ``pwsh`` → ``powershell`` → ``cmd``。

    为什么把 Git Bash 排这么前:指令模型写的几乎都是 POSIX 风格命令;而且实测
    单次启动 bash ≈ 60ms、cmd ≈ 85ms,而 pwsh ≈ 630ms、powershell 5.1 ≈ 406ms。
    """
    on_path = _find_on_path("bash")
    if on_path and not is_windows_app_alias(on_path):
        return "bash", [on_path, "-c"]

    native = _bash_shipped_with_git()
    if native:
        return "bash", [native, "-c"]

    for candidate in _well_known_bash_paths():
        if _file_exists(candidate):
            return "bash", [candidate, "-c"]

    fallbacks = (
        ("pwsh", "pwsh", ["-NoProfile", "-NonInteractive", "-Command"]),
        ("powershell", "powershell", ["-NoProfile", "-NonInteractive", "-Command"]),
        ("cmd", "cmd", ["/c"]),
    )
    for label, executable, tail in fallbacks:
        found = _find_on_path(executable)
        if found:
            return label, [found, *tail]
    return None


def detect_shell(prefer: str | None = None) -> tuple[str, list[str]]:
    """挑一个可用的 shell,返回 (标签, argv 前缀)。

    非 Windows 上就是 ``bash -c``(不是 ``-lc``):非登录 shell 直接继承父进程环境,
    既能拿到同样的 PATH,又不会去 source ``/etc/profile`` —— 那些脚本在某些机器上
    会往每次工具调用的输出里灌无关报错。
    """
    if prefer:
        return _shell_argv_for(prefer)

    if os.name == "nt":
        resolved = _resolve_windows_shell()
        if resolved is not None:
            return resolved
        raise RuntimeError(
            "找不到可用的 shell:bash(Git For Windows)、pwsh、powershell、cmd 都没有。"
            "可以用 --shell 或 MINI_HARNESS_SHELL 显式指定。"
        )

    return "bash", [_find_on_path("bash") or "/bin/bash", "-c"]


def _terminate(process: asyncio.subprocess.Process) -> None:
    """终止子进程。Windows 上 ``kill()`` 只杀直接子进程,bash 派生的孙子进程会留着,
    所以补一刀 ``taskkill /T`` 把整棵树带走。"""
    if process.returncode is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            process.kill()
    except OSError:
        pass


async def _collect_output(
    process: asyncio.subprocess.Process,
    timeout: float,
    cancellation: object,
) -> tuple[bytes, bool, bool]:
    """等子进程结束,同时与**超时**和**取消**赛跑。返回 (输出, 超时?, 被取消?)。

    这是"取消中断"落到工具层的样子:用户按 Ctrl+C 不该等命令自己跑完,
    而是立刻杀掉它并把"已中断"作为工具结果写回日志 —— 否则模型会以为命令失败了。
    """
    communicate = asyncio.ensure_future(process.communicate())
    waiters: set[asyncio.Future] = {communicate}
    cancel_waiter: asyncio.Future | None = None
    if cancellation is not None:
        cancel_waiter = asyncio.ensure_future(cancellation.wait())  # type: ignore[attr-defined]
        waiters.add(cancel_waiter)

    try:
        done, pending = await asyncio.wait(
            waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
    except asyncio.CancelledError:
        _terminate(process)
        for task in waiters:
            task.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
        raise
    for task in pending:
        task.cancel()

    timed_out = not done
    cancelled = (
        cancel_waiter is not None and cancel_waiter in done and communicate not in done
    )

    if timed_out or cancelled:
        _terminate(process)
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except (asyncio.TimeoutError, TimeoutError, ProcessLookupError):
            pass
        return b"", timed_out, cancelled

    raw, _ = communicate.result()
    return raw, False, False


#: 读进内存的安全上限 —— **不是**上下文预算。真正的预算由外溢策略在
#: ``tools/post-execute`` 上统一管(见 ``spill.py``),正常情况走不到这一支。
SAFETY_LIMIT_CHARS = 400_000


def make_shell_tool(
    default_cwd: Path | str | None = None,
    timeout: float = 60.0,
    max_output: int = SAFETY_LIMIT_CHARS,
    shell: str | None = None,
) -> Tool:
    """构造 shell 工具。超时与输出上限都是防止把上下文撑爆的护栏。"""
    label, argv_prefix = detect_shell(shell)
    child_env = _child_environment(label, argv_prefix[0])

    async def handler(args: dict, context: ToolCallContext) -> ToolResult:
        command = str(args.get("command") or "").strip()
        if not command:
            return ToolResult("缺少必填参数 command", is_error=True)

        workdir = Path(context.cwd or default_cwd or Path.cwd())
        process = await asyncio.create_subprocess_exec(
            *argv_prefix,
            command,
            cwd=str(workdir),
            env=child_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        raw, timed_out, cancelled = await _collect_output(
            process, timeout, context.cancellation
        )
        if cancelled:
            reason = getattr(context.cancellation, "reason", None) or "用户中断"
            return ToolResult(
                f"命令已被中断并终止({reason}): {command}", is_error=True
            )
        if timed_out:
            return ToolResult(
                f"命令超过 {timeout:.0f}s 未结束,已被终止: {command}", is_error=True
            )

        text = raw.decode("utf-8", errors="replace")
        if len(text) > max_output:
            # 走到这里说明外溢策略没生效(没装 spill 插件)。这里只保证不把内存读爆。
            text = (
                text[:max_output]
                + f"\n... [输出超过安全上限 {max_output} 字符,已硬截断;外溢策略未生效?]"
            )
        code = process.returncode
        return ToolResult(f"exit={code}\n{text}".rstrip(), is_error=code != 0)

    return Tool(
        name="shell",
        permission="execute",
        description=f"{DESCRIPTION}(当前后端:{label};默认工作目录为系统提示中的当前工作区，随工作区切换。)",
        parameters=SHELL_PARAMETERS,
        handler=handler,
    )


def plugin(
    cwd: Path | str | None = None,
    timeout: float = 60.0,
    shell: str | None = None,
) -> Plugin:
    """把 shell 工具注册进 ``ctx.tools``。"""

    def apply(ctx: Context) -> None:
        ctx.effect(ctx.tools.register(make_shell_tool(cwd, timeout, shell=shell)))

    return Plugin(
        name="tool-shell",
        apply=apply,
        inject=("tools",),
        description="本地 shell 执行工具",
    )
