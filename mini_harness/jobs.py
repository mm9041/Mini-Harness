"""Session-owned subprocess jobs; output goes to disk, never an unbounded pipe."""
from __future__ import annotations

import asyncio
import base64
import codecs
import json
import math
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .kernel import Context, Plugin
from .tools import Tool, ToolResult
from .builtin_tools.files import schema
from .builtin_tools.fs import _resolve


@dataclass
class Job:
    id: str
    owner: str
    command: str
    cwd: Path
    process: subprocess.Popen
    output_path: Path
    group: object | None = None
    sandbox_mode: str = "danger-full-access"
    started: float = field(default_factory=time.time)
    finished: float | None = None
    status: str = "running"
    done: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict:
        return {"job_id": self.id, "pid": self.process.pid, "command": self.command[:300], "sandbox_mode": self.sandbox_mode,
                "cwd": str(self.cwd), "status": self.status, "returncode": self.process.poll(),
                "duration_ms": int(((self.finished if self.finished is not None else time.time()) - self.started) * 1000),
                "output_path": str(self.output_path)}


class JobsService:
    def __init__(self, root: Path, max_running: int = 8, max_completed: int = 50):
        if isinstance(max_completed, bool) or not isinstance(max_completed, int) or max_completed < 1:
            raise ValueError("max_completed 必须为正整数")
        self.root = Path(root)
        self.max_running = max_running
        self.max_completed = max_completed
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        self.closed = False

    def start(self, command: str, cwd: Path, owner: str, timeout: float, *, permissions=None, mode="danger-full-access", workspace=None, session=None) -> Job:
        executable = shutil.which("pwsh") or (shutil.which("powershell") if os.name == "nt" else None)
        if executable is None:
            raise RuntimeError("没有可用的 PowerShell；请安装 pwsh 或使用兼容 shell 工具")
        confined = permissions is not None and mode != "danger-full-access"
        encoding = "" if confined else "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); $OutputEncoding=[Console]::OutputEncoding; "
        location = "Set-Location -LiteralPath '" + str(Path(cwd).resolve()).replace("'", "''") + "'; "
        code = "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; " + encoding + location + "& {\n" + command + "\n}; if ($null -ne $LASTEXITCODE) { exit $LASTEXITCODE }"
        argv = [executable, "-NoProfile", "-NonInteractive", "-OutputFormat", "Text", "-EncodedCommand", base64.b64encode(code.encode("utf-16-le")).decode("ascii")]
        if permissions is not None:
            argv = permissions.wrap_command(argv, mode=mode, workspace=workspace or cwd, session=session)
        with self.lock:
            if self.closed:
                raise RuntimeError("作业服务已关闭")
            if sum(not job.done.is_set() for job in self.jobs.values()) >= self.max_running:
                raise RuntimeError(f"同时运行的作业已达 {self.max_running} 个，请等待或终止现有作业")
            job_id = uuid.uuid4().hex
            folder = self.root / job_id
            folder.mkdir(parents=True)
            output = folder / "output.log"
            group = None
            if os.name == "nt":
                from .process_group import CREATE_SUSPENDED, WindowsProcessGroup
                group = WindowsProcessGroup()
            process = None
            try:
                with output.open("wb") as stream:
                    # DSH's Low-IL child must inherit a console. Give the runner a hidden
                    # console; CREATE_NO_WINDOW on the runner breaks child DLL initialization.
                    startup = None
                    flags = 0
                    if os.name == "nt":
                        flags = subprocess.CREATE_NO_WINDOW
                        if confined:
                            flags = subprocess.CREATE_NEW_CONSOLE
                            startup = subprocess.STARTUPINFO()
                            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                            startup.wShowWindow = 0
                        flags |= CREATE_SUSPENDED
                    process = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                               creationflags=flags, startupinfo=startup,
                                               start_new_session=os.name != "nt")
                if group:
                    group.assign_suspended(process)
            except BaseException:
                if process is not None:
                    process.kill()
                    process.wait()
                if group:
                    group.close()
                raise
            job = Job(job_id, owner, command, cwd, process, output, group=group, sandbox_mode=mode)
            self.jobs[job_id] = job
            threading.Thread(target=self._watch, args=(job, timeout), daemon=True, name=f"job-{job_id[:6]}").start()
            return job

    @staticmethod
    def _terminate(job: Job) -> None:
        if job.group is not None:
            job.group.terminate()
            return
        if job.process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(job.process.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            try:
                os.killpg(job.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if job.process.poll() is None:
            job.process.kill()

    def _watch(self, job: Job, timeout: float) -> None:
        try:
            try:
                job.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                with job.lock:
                    if job.status == "running":
                        job.status = "timed_out"
                        self._terminate(job)
                job.process.wait()
            with job.lock:
                if job.status == "running":
                    job.status = "completed" if job.process.returncode == 0 else "failed"
        finally:
            with job.lock:
                if job.group is not None:
                    job.group.close()
                elif os.name != "nt":
                    try:
                        os.killpg(job.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                job.finished = time.time()
                job.done.set()
            self._prune_completed()

    def _prune_completed(self) -> None:
        with self.lock:
            completed = sorted((job for job in self.jobs.values() if job.done.is_set()),
                               key=lambda job: job.finished if job.finished is not None else job.started,
                               reverse=True)
            for job in completed[self.max_completed:]:
                self.jobs.pop(job.id, None)
            # Keep output files: earlier tool responses may still reference their paths.

    def get(self, job_id: str, owner: str) -> Job:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or job.owner != owner:
                raise ValueError("找不到当前会话的作业")
            return job

    def list(self, owner: str) -> list[dict]:
        with self.lock:
            return [job.snapshot() for job in self.jobs.values() if job.owner == owner]

    def kill(self, job: Job) -> dict:
        with job.lock:
            if job.process.poll() is None:
                job.status = "killed"
                self._terminate(job)
        job.done.wait(5)
        return job.snapshot()

    def output(self, job: Job, offset: int = 0, limit: int = 20000) -> dict:
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset 必须是非负字节位置")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100000:
            raise ValueError("limit 必须为 1..100000 字节")
        with job.output_path.open("rb") as stream:
            stream.seek(offset)
            raw = stream.read(limit)
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            text = decoder.decode(raw)
            # Finish a trailing UTF-8 character without skipping its remaining bytes.
            for _ in range(3):
                if not decoder.getstate()[0]:
                    break
                extra = stream.read(1)
                if not extra:
                    break
                raw += extra
                text += decoder.decode(extra)
            pending = len(decoder.getstate()[0])
            if job.done.is_set():
                text += decoder.decode(b"", final=True)
                pending = 0
        return {**job.snapshot(), "output": text,
                "next_offset": offset + len(raw) - pending, "total_bytes": job.output_path.stat().st_size}

    def close(self) -> None:
        with self.lock:
            self.closed = True
            jobs = list(self.jobs.values())
        for job in jobs:
            if not job.done.is_set():
                self.kill(job)


def _timeout(value, default: float, *, waiting: bool = False) -> float:
    message = "等待 timeout 必须为 0..60 秒" if waiting else "timeout 必须大于 0 秒且不超过 86400 秒"
    value = default if value is None else value
    try:
        if isinstance(value, bool):
            raise ValueError(message)
        seconds = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(seconds) or not (0 <= seconds <= 60 if waiting else 0 < seconds <= 86400):
        raise ValueError(message)
    return seconds


def plugin(root: Path, default_timeout: float = 60, allow_outside=False) -> Plugin:
    def apply(ctx: Context):
        service = JobsService(root)
        ctx.provide("jobs", service)
        ctx.effect(service.close)

        def owner(context):
            return context.session.id if context.session else "standalone"

        async def pwsh(args, context):
            command = args.get("command")
            if not isinstance(command, str) or not command.strip():
                raise ValueError("command 不能为空")
            timeout = _timeout(args.get("timeout"), default_timeout)
            background = args.get("background", False)
            if not isinstance(background, bool):
                raise ValueError("background 必须是布尔值")
            cwd = _resolve(str(args.get("cwd") or "."), context, allow_outside)
            if not cwd.is_dir():
                raise ValueError("cwd 不是有效目录")
            if context.cancellation and context.cancellation.cancelled:
                return ToolResult("已取消，未启动命令", True)
            # Start synchronously so task cancellation cannot orphan a partially-created job.
            permissions = ctx.get("permissions", None)
            job = service.start(command, cwd, owner(context), timeout, permissions=permissions,
                                mode=context.sandbox_mode or "danger-full-access", workspace=context.cwd, session=context.session)
            if background:
                return ToolResult(json.dumps(job.snapshot(), ensure_ascii=False))
            try:
                while not job.done.is_set():
                    if context.cancellation and context.cancellation.cancelled:
                        await asyncio.to_thread(service.kill, job)
                        break
                    await asyncio.sleep(.05)
            except asyncio.CancelledError:
                await asyncio.to_thread(service.kill, job)
                raise
            result = await asyncio.to_thread(service.output, job)
            return ToolResult(json.dumps(result, ensure_ascii=False), job.status != "completed")

        async def job_list(args, context):
            return ToolResult(json.dumps(service.list(owner(context)), ensure_ascii=False))

        async def job_output(args, context):
            job = service.get(str(args.get("job_id") or ""), owner(context))
            if args.get("wait", False):
                wait_seconds = _timeout(args.get("timeout"), 30, waiting=True)
                until = time.monotonic() + wait_seconds
                while not job.done.is_set() and time.monotonic() < until:
                    if context.cancellation and context.cancellation.cancelled:
                        return ToolResult("等待已取消；后台作业仍运行，可用 job_kill 终止。", True)
                    await asyncio.sleep(.05)
            result = await asyncio.to_thread(service.output, job, args.get("offset", 0), args.get("limit", 20000))
            return ToolResult(json.dumps(result, ensure_ascii=False))

        async def job_kill(args, context):
            job = service.get(str(args.get("job_id") or ""), owner(context))
            return ToolResult(json.dumps(await asyncio.to_thread(service.kill, job), ensure_ascii=False))

        definitions = [
            Tool("pwsh", "执行 PowerShell 命令。支持 cwd、秒级 timeout 和 background。后台作业跨轮次运行，退出 harness 时终止。", schema({"command": {"type": "string"}, "cwd": {"type": "string"}, "timeout": {"type": "number"}, "background": {"type": "boolean"}}, ("command",)), pwsh, permission="execute", sandboxed=True),
            Tool("job_list", "列出当前会话的全部运行中作业及最近 50 条已结束作业。更早的作业记录会移除，输出文件仍保留。", schema({}, ()), job_list, True, permission="read"),
            Tool("job_output", "读取作业 UTF-8 输出。offset/limit 以字节为单位，可用 next_offset 继续读取。wait 超时仅结束等待，不终止作业。", schema({"job_id": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}, "wait": {"type": "boolean"}, "timeout": {"type": "number"}}, ("job_id",)), job_output, permission="read"),
            Tool("job_kill", "终止当前会话创建的作业及子进程树，不接受任意系统 PID。", schema({"job_id": {"type": "string"}}, ("job_id",)), job_kill, permission="process-control"),
        ]
        for tool in definitions:
            ctx.effect(ctx.tools.register(tool))
    return Plugin("tool-jobs", apply, inject=("tools",), description="PowerShell 与后台作业")
