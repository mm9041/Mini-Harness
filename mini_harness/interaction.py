"""Questions that suspend a tool call, and persistent file deliverables."""
from __future__ import annotations

import asyncio
import inspect
import json
import mimetypes
import math
import uuid
from pathlib import Path

from .builtin_tools.files import schema
from .builtin_tools.fs import _resolve
from .kernel import Plugin
from .tools import Tool, ToolResult


class UserQuestions:
    def __init__(self, ctx, timeout=600):
        self.ctx = ctx
        self.pending = {}
        self.responder = None  # Blocking input must use an async responder.
        self.browser = False
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("提问超时必须为正的有限秒数")
        self.timeout = timeout

    def snapshot(self):
        return [entry[0] for entry in self.pending.values()]

    def answer(self, question_id, answer, skipped=False):
        entry = self.pending.get(question_id)
        if not entry or entry[1].done():
            return False
        if not isinstance(answer, str) or len(answer) > 10000 or (not skipped and not answer.strip()):
            raise ValueError("回答不能为空，且不能超过 10000 字符")
        entry[1].set_result({"answer": answer.strip(), "skipped": bool(skipped)})
        return True

    async def ask(self, args, context):
        question, options = args.get("question"), args.get("options", [])
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ValueError("question 必须为 1..2000 字符的问题")
        if not isinstance(options, list) or len(options) > 6 or any(not isinstance(o, str) or not o.strip() or len(o) > 200 for o in options):
            raise ValueError("options 最多包含 6 个选项，每项为 1..200 字符")
        if not self.browser and self.responder is None:
            return ToolResult("当前没有可用的交互界面，请在最终回复中说明需要用户补充的信息。", True)
        if context.cancellation and context.cancellation.cancelled:
            return ToolResult("用户已停止，问题未发出", True)
        question_id = uuid.uuid4().hex
        message = {"kind": "question", "id": question_id, "question": question.strip(), "options": options}
        future = asyncio.get_running_loop().create_future()
        self.pending[question_id] = (message, future)
        if context.session:
            context.session.append("interaction/question", **message)
        answer = {"answer": "", "skipped": True, "reason": "问题已取消"}
        cancel = asyncio.create_task(context.cancellation.wait()) if context.cancellation else None
        responding = None
        try:
            await self.ctx.emit("interaction/question", context.session, message)
            if not self.browser and self.responder:
                async def respond():
                    result = self.responder(message)
                    if inspect.isawaitable(result):
                        result = await result
                    if not self.answer(question_id, result or "", skipped=result is None):
                        # First accepted answer wins; never replace it with a late response.
                        return
                responding = asyncio.create_task(respond())
            waiting = {task for task in (future, cancel, responding) if task is not None}
            done, _ = await asyncio.wait(waiting, timeout=self.timeout, return_when=asyncio.FIRST_COMPLETED)
            if future in done:
                answer = future.result()
            elif cancel and cancel in done:
                answer["reason"] = "用户已停止"
            elif responding and responding in done:
                try:
                    responding.result()
                except Exception as exc:
                    answer["reason"] = f"应答界面出错：{type(exc).__name__}"
                    raise  # Preserve the original error while recording an accurate close reason.
            else:
                answer["reason"] = "等待回答超时"
            return ToolResult(json.dumps(answer, ensure_ascii=False), answer["skipped"])
        finally:
            self.pending.pop(question_id, None)
            if not future.done():
                future.cancel()
            helpers = [task for task in (cancel, responding) if task is not None]
            for task in helpers:
                task.cancel()
            await asyncio.gather(*helpers, return_exceptions=True)
            closed = {"kind": "question-closed", "id": question_id, **answer}
            if context.session:
                context.session.append("interaction/answer", **closed)
            await self.ctx.emit("interaction/question", context.session, closed)


def preview_type(path):
    ext = path.suffix.lower()
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
        return "image", mimetypes.guess_type(path.name)[0] or "image/png"
    if ext == ".pdf":
        return "pdf", "application/pdf"
    if ext in (".txt", ".md", ".csv", ".tsv", ".json", ".yaml", ".yml", ".xml", ".html", ".htm", ".svg", ".py", ".js", ".ts", ".css", ".log", ".tex"):
        return "text", "text/plain; charset=utf-8"
    return "download", "application/octet-stream"


def plugin(allow_outside=False, *, question_timeout=600):
    def apply(ctx):
        questions = UserQuestions(ctx, timeout=question_timeout)
        ctx.provide("userQuestions", questions)

        async def present(args, context):
            files = args.get("files")
            if not context.session:
                raise ValueError("交付物需要关联当前会话")
            if not isinstance(files, list) or not 1 <= len(files) <= 10:
                raise ValueError("files 必须包含 1..10 个文件")
            artifacts = []
            for item in files:
                if not isinstance(item, dict) or not isinstance(item.get("file_path"), str) or not item["file_path"].strip():
                    raise ValueError("每个交付物需要 file_path")
                path = _resolve(item["file_path"], context, allow_outside)
                if not path.is_file():
                    raise ValueError(f"交付文件不存在或不是普通文件: {path}")
                title = item.get("title", path.name)
                if not isinstance(title, str) or not title.strip() or len(title) > 200:
                    raise ValueError("交付物标题必须为 1..200 字符")
                kind, mime = preview_type(path)
                artifacts.append({"id": uuid.uuid4().hex, "name": path.name, "title": title,
                                  "path": path.as_posix(), "root": (path.parent if allow_outside else context.cwd.resolve()).as_posix(),
                                  "size": path.stat().st_size, "preview": kind, "mime": mime})
            if context.cancellation and context.cancellation.cancelled:
                return ToolResult("用户已停止，未标记交付物", True)
            for artifact in artifacts:
                context.session.append("artifact/presented", **artifact)
                await ctx.emit("artifact/presented", context.session, artifact)
            return ToolResult(json.dumps({"artifacts": artifacts, "note": "已标记为最终交付物，文件内容仍以磁盘上的最新版本为准。"}, ensure_ascii=False))

        ctx.effect(ctx.tools.register(Tool("ask_user_question", "需要补充信息或用户确认时提出一个明确问题，暂停当前步骤并等待回答。可给出选项，用户也可自由输入。跳过或取消不代表同意。", schema({"question": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}, "maxItems": 6}}, ("question",)), questions.ask, permission="interact")))
        ctx.effect(ctx.tools.register(Tool("present", "将已经存在的文件标记为最终交付物，生成可预览、打开或下载的卡片。先完成并验证文件，再调用此工具；它不会创建文件。", schema({"files": {"type": "array", "minItems": 1, "maxItems": 10, "items": schema({"file_path": {"type": "string"}, "title": {"type": "string"}}, ("file_path",))}}, ("files",)), present, permission="read")))
    return Plugin("tool-interaction", apply, inject=("tools",), description="用户提问与文件交付")
