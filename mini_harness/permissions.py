"""DSH-style permission presets: per-session policy, one-call escalation, real runners."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import re
from dataclasses import dataclass
from pathlib import Path

from .approval import ApprovalRequest
from .kernel import Plugin
from .tools import ToolResult


@dataclass(frozen=True)
class PermissionPreset:
    sandbox: str
    approval: str
    label: str


PRESETS = {
    "read-only": PermissionPreset("read-only", "ask", "只读"),
    "workspace-write": PermissionPreset("workspace-write", "ask", "工作区内修改"),
    "danger-full-access": PermissionPreset("danger-full-access", "never", "完全访问"),
}
RANK = {"read-only": 0, "workspace-write": 1, "danger-full-access": 2}


def grant_private_temp(path):
    """Only for freshly created private directories, not a user's workspace ACL."""
    if os.name == "nt":
        raw = subprocess.check_output(["whoami", "/user", "/fo", "csv", "/nh"], creationflags=subprocess.CREATE_NO_WINDOW)
        sid = re.search(rb"S-1-5-21-[0-9-]+", raw)
        if not sid:
            raise RuntimeError("SANDBOX_UNAVAILABLE: 无法获取临时目录所有者 SID")
        subprocess.run(["icacls", str(path), "/grant", f"*{sid.group().decode()}:(OI)(CI)F", "/Q"],
                       capture_output=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)


class PermissionService:
    def __init__(self, ctx, default="workspace-write", runtime_root=None):
        if default not in PRESETS:
            raise ValueError(f"未知权限预设 {default!r}")
        self.ctx, self.default = ctx, default
        self.runtime_root = Path(runtime_root) if runtime_root else Path.home() / ".mini-harness" / "sandbox-runtime"
        self._temps = {}

    def current(self, session=None):
        if session is not None:
            parent = self.ctx.approval.parent_session(session)
            if parent is not None:
                return self.current(parent)
            for event in reversed(session.events):
                if event.type == "permission/preset":
                    value = event.data.get("preset")
                    if value not in PRESETS:
                        raise ValueError("会话权限预设无效，拒绝执行")
                    return value
            for event in reversed(session.events):
                if event.type == "approval/policy":
                    # Old auto-approval is not evidence of permission for unrestricted FS.
                    return "read-only" if event.data.get("mode") == "deny" else "workspace-write"
        return self.default

    def pin(self, session):
        if not session.events_of("permission/preset"):
            value = self.current(session)
            session.append("permission/preset", preset=value,
                           sandbox=PRESETS[value].sandbox, approval=PRESETS[value].approval)

    def set(self, preset, session):
        if preset not in PRESETS:
            raise ValueError(f"未知权限预设 {preset!r}")
        previous = self.current(session)
        if preset != previous or not session.events_of("permission/preset"):
            self.ctx.approval.invalidate()
            session.append("permission/preset", preset=preset,
                           sandbox=PRESETS[preset].sandbox, approval=PRESETS[preset].approval)
        return previous

    def snapshot(self, session=None):
        value = self.current(session)
        preset = PRESETS[value]
        return {"preset": value, "sandbox": preset.sandbox, "approval": preset.approval,
                "options": {key: spec.label for key, spec in PRESETS.items()},
                "backend": "dsh-windows-acl" if os.name == "nt" else "unavailable",
                "read_isolation": False, "network_isolation": False}

    def temp_dir(self, session, workspace):
        key = (session.id if session else "standalone", str(Path(workspace).resolve()))
        if key not in self._temps:
            root = Path(tempfile.gettempdir()).resolve()
            workspace = Path(workspace).resolve()
            if root == workspace or workspace in root.parents:
                raise RuntimeError("SANDBOX_UNAVAILABLE: 临时目录必须位于工作区之外")
            private = Path(tempfile.mkdtemp(prefix="mini-sandbox-", dir=root)).resolve()
            try:
                grant_private_temp(private)
            except Exception:
                shutil.rmtree(private, ignore_errors=True)
                raise
            self._temps[key] = private
        return self._temps[key]

    def resolve_path(self, raw, context, writing=False):
        path = Path(raw).expanduser()
        target = (context.cwd / path).resolve() if not path.is_absolute() else path.resolve()
        mode = context.sandbox_mode or self.current(context.session)
        if not writing or mode == "danger-full-access":
            return target
        if mode == "workspace-write":
            roots = (context.cwd.resolve(), self.temp_dir(context.session, context.cwd))
            if any(target == root or root in target.parents for root in roots):
                return target
        raise ValueError(f"SANDBOX_DENIED: {mode} 禁止写入 {target}；可为本次调用指定 sandbox_permissions 和 justification 请求扩大权限")

    async def authorize(self, call, context):
        if context.session is not None:
            self.pin(context.session)
        standing = self.current(context.session)
        context.sandbox_mode = standing
        requested = call.arguments.get("sandbox_permissions")
        if requested is not None:
            if requested not in ("workspace-write", "danger-full-access"):
                return ToolResult("sandbox_permissions 必须为 workspace-write 或 danger-full-access", True)
            if RANK[requested] > RANK[standing]:
                reason = call.arguments.get("justification")
                if not isinstance(reason, str) or not reason.strip():
                    return ToolResult("扩大权限必须提供 justification", True)
                decision = await self.ctx.approval.decide(ApprovalRequest(
                    call, f"本次调用申请 {standing} → {requested}：{reason}",
                    session=context.session, cancellation=context.cancellation,
                    policy=PRESETS[standing].approval))
                if not decision.approved:
                    return ToolResult(f"权限申请被拒绝：{decision.note}", True)
                context.sandbox_mode = requested
        permission = self.ctx.tools.permission_for(call.name)
        if context.sandbox_mode == "danger-full-access":
            return None
        # Only capabilities that actually implement the policy may run arbitrary writes/code.
        if permission in {"read", "network", "delegate", "interact", "process-control"}:
            return None
        if not self.ctx.tools.is_sandboxed(call.name):
            return ToolResult("SANDBOX_DENIED: 此工具没有沙箱执行后端；需显式申请本次 danger-full-access", True)
        if permission == "write":
            try:
                self.resolve_path(str(call.arguments.get("file_path", call.arguments.get("path", ""))), context, True)
            except ValueError as exc:
                return ToolResult(str(exc), True)
        return None

    def wrap_command(self, argv, *, mode, workspace, session=None):
        if mode == "danger-full-access":
            return list(argv)
        if mode not in PRESETS:
            raise ValueError("无效沙箱模式")
        if os.name != "nt":
            raise RuntimeError("SANDBOX_UNAVAILABLE: 当前仅接入 Windows ACL 命令沙箱")
        node = shutil.which("node")
        runner = self.runtime_root / "sandbox_runner.cjs"
        backend = self.runtime_root / "node_modules/@deepseek-ai/dsh-sandbox-windows-acl/lib/runner.js"
        if not node or not runner.is_file() or not backend.is_file():
            raise RuntimeError("SANDBOX_UNAVAILABLE: 请安装 Node.js 并执行 python scripts/setup_sandbox.py")
        root = Path(workspace).resolve()
        if any(root == path or root in path.parents for path in (runner.resolve(), backend.resolve(), Path(node).resolve())):
            raise RuntimeError("SANDBOX_UNAVAILABLE: 沙箱运行时和 Node 必须位于可写工作区之外")
        private_temp = self.temp_dir(session, workspace)
        if mode == "workspace-write":
            from .windows_acl import prepare_workspace_acl
            repair = prepare_workspace_acl(root, self.runtime_root / "acl-backups")
            if repair and session is not None:
                session.append("permission/workspace-acl-repaired", **repair)
        return [node, str(runner), "--workspace", str(Path(workspace).resolve()),
                "--temp", str(private_temp), "--mode", mode, "--", *argv]

    def close(self):
        jobs = self.ctx.get("jobs", None) if self.ctx else None
        if jobs is not None:
            jobs.close()
        for path in self._temps.values():
            shutil.rmtree(path, ignore_errors=True)
        self._temps.clear()


def plugin(default="workspace-write", runtime_root=None):
    def apply(ctx):
        service = PermissionService(ctx, default, runtime_root)
        ctx.provide("permissions", service)
        ctx.effect(service.close)
    return Plugin("permissions", apply, inject=("approval", "tools"), description="沙箱与审批权限预设")
