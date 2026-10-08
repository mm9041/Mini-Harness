"""``read_file`` / ``write_file`` —— 第二个(和第三个)面向模型的工具。

对应 dsh 的 ``packages/fs/tool-fs``(它的工具名是 ``read`` / ``write`` / ``edit``)。
两个工具都很小,但把"工具 = 插件"这件事的第二面展示出来了:**策略可以和能力分开**。

* `write_file` 动了用户的工作区,所以默认**总是需要审批**(策略在 ``approval.py``
  里配,工具本身不写审批逻辑);
* 两个工具都限制在**工作区之内**:相对路径按工作目录解析,越界直接拒绝 ——
  想放开就显式 `allow_outside=True`。
"""

from __future__ import annotations

import asyncio

from pathlib import Path

from ..kernel import Context, Plugin
from ..tools import Tool, ToolCallContext, ToolResult

__all__ = [
    "READ_PARAMETERS",
    "WRITE_PARAMETERS",
    "make_read_tool",
    "make_write_tool",
    "plugin",
]

#: 读进内存的安全上限 —— **不是**上下文预算。真正的预算由外溢策略在
#: ``tools/post-execute`` 上统一管(见 ``spill.py``)。
DEFAULT_MAX_OUTPUT = 400_000
DEFAULT_READ_LINES = 2000

READ_PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "文件路径,相对工作目录或绝对路径"},
        "offset": {
            "type": "integer",
            "description": "从第几行开始读(1 起,默认 1)",
            "minimum": 1,
        },
        "limit": {
            "type": "integer",
            "description": f"最多读多少行(默认 {DEFAULT_READ_LINES})",
            "minimum": 1,
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}

WRITE_PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "文件路径,相对工作目录或绝对路径"},
        "content": {"type": "string", "description": "完整文件内容(UTF-8,整体覆盖)"},
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}


def _resolve(
    raw: str,
    context: ToolCallContext,
    allow_outside: bool,
    extra_roots: tuple[Path | str, ...] = (),
    writing: bool = False,
) -> Path:
    """把模型给的路径解析成绝对路径,并限制在允许的范围内。

    ``extra_roots`` 是**只读白名单**:外溢目录(``spill.py`` 落的那些文件)在工作区
    之外,但模型必须能把它们读回来。这体现一条策略:**读可以放宽,写仍然锁死**。
    """
    permissions = context.ctx.get("permissions", None) if context.ctx else None
    if permissions is not None:
        return permissions.resolve_path(raw, context, writing)
    candidate = Path(raw).expanduser()
    base = Path(context.cwd or Path.cwd())
    target = (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

    if allow_outside:
        return target

    workspace = base.resolve()
    if target == workspace or workspace in target.parents:
        return target

    for root in extra_roots:
        root_path = Path(root).expanduser().resolve()
        if target == root_path or root_path in target.parents:
            return target

    raise ValueError(
        f"路径越界:{target} 不在工作区 {workspace} 之内"
        "(外溢目录可读;要访问其他位置请显式打开 allow_outside)"
    )


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return (
        f"{text[:limit]}\n... [超过安全上限 {limit} 字符,已硬截断;外溢策略未生效?]"
    )


def make_read_tool(
    default_cwd: Path | str | None = None,
    allow_outside: bool = False,
    max_output: int = DEFAULT_MAX_OUTPUT,
    extra_roots: tuple[Path | str, ...] = (),
) -> Tool:
    async def handler(args: dict, context: ToolCallContext) -> ToolResult:
        raw = str(args.get("path") or "").strip()
        if not raw:
            return ToolResult("缺少必填参数 path", is_error=True)

        try:
            target = _resolve(raw, context, allow_outside, extra_roots)
        except ValueError as exc:
            return ToolResult(str(exc), is_error=True)

        if not target.exists():
            return ToolResult(f"文件不存在: {target}", is_error=True)
        if target.is_dir():
            return ToolResult(f"{target} 是目录,不是文件", is_error=True)

        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult(f"读取失败:{exc}", is_error=True)

        lines = text.splitlines()
        offset = max(1, int(args.get("offset") or 1))
        limit = max(1, int(args.get("limit") or DEFAULT_READ_LINES))
        window = lines[offset - 1 : offset - 1 + limit]

        header = (
            f"# {target}\n# 共 {len(lines)} 行,显示第 {offset}-{offset + len(window) - 1} 行\n"
        )
        return ToolResult(_truncate(header + "\n".join(window), max_output))

    return Tool(
        name="read_file",
        permission="read",
        safe_to_prefetch=True,
        description=(
            "读取一个文本文件(UTF-8),可按行区间读取。"
            "只能访问工作区内的路径。"
        ),
        parameters=READ_PARAMETERS,
        handler=handler,
    )


def make_write_tool(
    default_cwd: Path | str | None = None,
    allow_outside: bool = False,
) -> Tool:
    async def handler(args: dict, context: ToolCallContext) -> ToolResult:
        raw = str(args.get("path") or "").strip()
        if not raw:
            return ToolResult("缺少必填参数 path", is_error=True)
        if "content" not in args:
            return ToolResult("缺少必填参数 content", is_error=True)

        try:
            target = _resolve(raw, context, allow_outside, writing=True)
        except ValueError as exc:
            return ToolResult(str(exc), is_error=True)

        if target.is_dir():
            return ToolResult(f"{target} 是目录,不能写入", is_error=True)

        content = str(args.get("content") or "")
        previous = target.stat().st_size if target.exists() else None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            from .files import atomic_write
            await asyncio.to_thread(atomic_write, target, content)
        except OSError as exc:
            return ToolResult(f"写入失败:{exc}", is_error=True)

        written = target.stat().st_size
        if previous is None:
            return ToolResult(f"已创建 {target}({written} 字节)")
        return ToolResult(f"已覆盖 {target}({previous} → {written} 字节)")

    return Tool(
        name="write_file",
        permission="write",
        sandboxed=True,
        description=(
            "把内容整体写入文件(UTF-8,已存在则覆盖),父目录会自动创建。"
            "只能访问工作区内的路径;该工具默认需要人工审批。"
        ),
        parameters=WRITE_PARAMETERS,
        handler=handler,
    )


def plugin(
    cwd: Path | str | None = None,
    allow_outside: bool = False,
    max_output: int = DEFAULT_MAX_OUTPUT,
    extra_roots: tuple[Path | str, ...] = (),
) -> Plugin:
    """注册 ``read_file`` 与 ``write_file``。

    ``extra_roots`` 只给 ``read_file`` —— 外溢目录应当可读,但不该可写。
    """

    def apply(ctx: Context) -> None:
        ctx.effect(
            ctx.tools.register(
                make_read_tool(cwd, allow_outside, max_output, extra_roots)
            )
        )
        ctx.effect(ctx.tools.register(make_write_tool(cwd, allow_outside)))

    return Plugin(
        name="tool-fs",
        apply=apply,
        inject=("tools",),
        description="文件读写工具(工作区限定)",
    )
