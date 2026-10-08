"""Canonical filesystem tools, sharing the existing workspace resolver."""
from __future__ import annotations

import asyncio
import base64
import io
import os
import re
import stat
import tempfile
from pathlib import Path

from ..kernel import Context, Plugin
from ..tools import Tool, ToolResult
from .fs import _resolve, make_write_tool


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".harness-edit-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def schema(properties: dict, required: tuple[str, ...]) -> dict:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


PATH = {"type": "string", "description": "文件路径，相对当前工作区或绝对路径"}


def file_path(args: dict) -> str:
    value = args.get("file_path", args.get("path"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("缺少 file_path")
    return value


def bounded(value, default: int, maximum: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("offset/limit 必须是正整数")
    return min(value, maximum)


def plugin(allow_outside=False, extra_roots=()) -> Plugin:
    def apply(ctx: Context):
        def resolve(args, context, reading=True):
            return _resolve(file_path(args), context, allow_outside, extra_roots if reading else (), writing=not reading)

        def read_sync(args, context):
            path = resolve(args, context)
            offset = bounded(args.get("offset"), 1, 2**31)
            limit = bounded(args.get("limit"), 2000, 10000)
            rows, chars, total, clipped = [], 0, 0, False
            with path.open(encoding="utf-8-sig", errors="replace", newline="") as stream:
                for total, line in enumerate(stream, 1):
                    if offset <= total < offset + limit:
                        row = f"{total:>6}\t{line.rstrip(chr(10)).rstrip(chr(13))}"
                        if chars + len(row) <= 400_000:
                            rows.append(row)
                            chars += len(row)
                        else:
                            clipped = True
            footer = f"\n[共 {total} 行；offset={offset}；返回 {len(rows)} 行]"
            if clipped:
                footer += "\n[输出达到上限，请缩小 limit 或用 grep 检索]"
            return ToolResult(f"# {path}\n" + "\n".join(rows) + footer)

        async def read(args, context):
            return await asyncio.to_thread(read_sync, args, context)

        original_write = make_write_tool(allow_outside=allow_outside)

        async def write(args, context):
            if not isinstance(args.get("content"), str):
                raise ValueError("content 必须是字符串")
            return await original_write.handler({**args, "path": file_path(args)}, context)

        def edit_sync(args, context):
            path = resolve(args, context, False)
            old, new = args.get("old_string"), args.get("new_string")
            if not isinstance(old, str) or not old:
                raise ValueError("old_string 必须是非空字面文本")
            if not isinstance(new, str):
                raise ValueError("new_string 必须是字符串，可为空")
            replace_all = args.get("replace_all", False)
            if not isinstance(replace_all, bool):
                raise ValueError("replace_all 必须是布尔值")
            content = path.read_bytes().decode("utf-8")
            count = content.count(old)
            if count == 0:
                return ToolResult("没有找到 old_string；请先 read 核对原文（包括空格与换行）。", True)
            if count > 1 and not replace_all:
                return ToolResult(f"找到 {count} 处匹配，未修改。请提供更长的唯一片段，或指定 replace_all=true。", True)
            atomic_write(path, content.replace(old, new, -1 if replace_all else 1))
            return ToolResult(f"已编辑 {path}，替换 {count if replace_all else 1} 处。")

        async def edit(args, context):
            return await asyncio.to_thread(edit_sync, args, context)

        def search_root(args, context):
            path = _resolve(str(args.get("path") or "."), context, allow_outside, extra_roots)
            if not path.exists():
                raise ValueError("检索路径不存在")
            return path

        def files(root, pattern, context):
            if not isinstance(pattern, str) or not pattern or Path(pattern).is_absolute() or ".." in Path(pattern).parts:
                raise ValueError("glob 必须是工作区内的相对路径模式")
            candidates = [root] if root.is_file() else root.glob(pattern)
            for path in candidates:
                try:
                    _resolve(str(path), context, allow_outside, extra_roots)
                    if path.is_file():
                        yield path
                except (OSError, ValueError):
                    continue

        def glob_sync(args, context):
            root = search_root(args, context)
            limit = bounded(args.get("limit"), 200, 5000)
            found = []
            for path in files(root, args.get("pattern", "**/*"), context):
                found.append(str(path.relative_to(root) if root.is_dir() else path.name))
                if len(found) > limit:
                    break
            return ToolResult("\n".join(sorted(found[:limit])) + ("\n[达到 limit，请缩小模式]" if len(found) > limit else "") or "未找到文件")

        def grep_sync(args, context):
            root = search_root(args, context)
            pattern = args.get("pattern")
            if not isinstance(pattern, str):
                raise ValueError("pattern 必须是正则表达式字符串")
            regex = re.compile(pattern, re.IGNORECASE if args.get("ignore_case", False) else 0)
            limit = bounded(args.get("limit"), 200, 5000)
            matches, skipped = [], 0
            for path in files(root, args.get("glob", "**/*"), context):
                if context.cancellation and context.cancellation.cancelled:
                    return ToolResult("检索已取消", True)
                try:
                    with path.open("rb") as stream:
                        if b"\x00" in stream.read(8192):
                            continue
                    with path.open(encoding="utf-8-sig", errors="replace") as stream:
                        for number, line in enumerate(stream, 1):
                            if regex.search(line):
                                label = path.relative_to(root) if root.is_dir() else path.name
                                matches.append(f"{label}:{number}:{line.rstrip()[:3000]}")
                                if len(matches) >= limit:
                                    return ToolResult("\n".join(matches) + "\n[达到 limit，可缩小路径或模式后继续检索]")
                except OSError:
                    skipped += 1
            return ToolResult(("\n".join(matches) or "无匹配") + (f"\n[跳过 {skipped} 个无法读取的文件]" if skipped else ""))

        async def glob(args, context):
            return await asyncio.to_thread(glob_sync, args, context)

        async def grep(args, context):
            return await asyncio.to_thread(grep_sync, args, context)

        def image_sync(args, context):
            try:
                from PIL import Image, ImageOps
            except ImportError as exc:
                raise RuntimeError('read_image 需要 Pillow；请安装 mini-harness[images] 或 Pillow') from exc
            path = resolve(args, context)
            if path.stat().st_size > 30 * 1024 * 1024:
                raise ValueError("图片文件超过 30MB，请先压缩")
            edge = bounded(args.get("max_edge"), 1600, 2048)
            with Image.open(path) as source:
                if source.format not in ("PNG", "JPEG", "WEBP", "GIF"):
                    raise ValueError("仅支持 PNG/JPEG/WebP/GIF")
                original = source.size
                if original[0] * original[1] > 40_000_000:
                    raise ValueError("图片超过 4000 万像素，请先缩小")
                animated = getattr(source, "n_frames", 1) > 1
                source.seek(0)
                picture = ImageOps.exif_transpose(source).convert("RGBA")
                picture.thumbnail((edge, edge), Image.Resampling.LANCZOS)
                output = io.BytesIO()
                picture.save(output, format="PNG")
                image = {"data_url": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii"),
                         "width": picture.width, "height": picture.height, "path": str(path)}
            text = f"已读取 {path}：{original[0]}×{original[1]} → {image['width']}×{image['height']}"
            if animated:
                text += "（动态图读取第一帧）"
            return ToolResult(text, images=[image])

        async def read_image(args, context):
            return await asyncio.to_thread(image_sync, args, context)

        definitions = [
            Tool("read", "读取 UTF-8 文本，带行号与分页。", schema({"file_path": PATH, "offset": {"type": "integer", "minimum": 1}, "limit": {"type": "integer", "minimum": 1}}, ("file_path",)), read, True, permission="read"),
            Tool("write", "创建或原子覆盖 UTF-8 文件。修改局部代码优先使用 edit。", schema({"file_path": PATH, "content": {"type": "string"}}, ("file_path", "content")), write, permission="write", sandboxed=True),
            Tool("edit", "按字面文本精确替换。默认要求唯一匹配；replace_all 替换全部。", schema({"file_path": PATH, "old_string": {"type": "string"}, "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}}, ("file_path", "old_string", "new_string")), edit, permission="write", sandboxed=True),
            Tool("glob", "按相对路径模式查找文件，包含隐藏文件和被 Git 忽略的文件；如 **/*.py。", schema({"pattern": {"type": "string"}, "path": PATH, "limit": {"type": "integer"}}, ("pattern",)), glob, True, permission="read"),
            Tool("grep", "按正则检索 UTF-8 文件内容，返回路径、行号和匹配行；包含隐藏/忽略文件，跳过二进制。", schema({"pattern": {"type": "string"}, "path": PATH, "glob": {"type": "string"}, "ignore_case": {"type": "boolean"}, "limit": {"type": "integer"}}, ("pattern",)), grep, True, permission="read"),
            Tool("read_image", "读取 PNG/JPEG/WebP/GIF 并作为图像交给视觉模型；大图自动缩放，GIF 取首帧。", schema({"file_path": PATH, "max_edge": {"type": "integer", "minimum": 1, "maximum": 2048}}, ("file_path",)), read_image, True, permission="read"),
        ]
        for tool in definitions:
            ctx.effect(ctx.tools.register(tool))

    return Plugin("tool-files", apply, inject=("tools",), description="精确编辑、文件检索与图片读取")
