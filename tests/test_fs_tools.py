"""``read_file`` / ``write_file`` 测试 —— 含工作区越界防护。"""

from __future__ import annotations

import unittest

from mini_harness.adapters.mock import scripted_plugin
from mini_harness.builtin_tools.fs import make_read_tool, make_write_tool
from mini_harness.tools import ToolCallContext

from . import support
from .support import build_context_with as build_test_context


class _ToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()
        self.context = ToolCallContext(call_id="c", cwd=self.cwd)

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    def write(self, name: str, content: str) -> str:
        target = self.cwd / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return name


class ReadFileTests(_ToolTests):
    async def test_reads_the_whole_file(self) -> None:
        self.write("a.txt", "第一行\n第二行\n第三行")
        tool = make_read_tool(self.cwd)

        result = await tool.handler({"path": "a.txt"}, self.context)

        self.assertFalse(result.is_error)
        self.assertIn("共 3 行", result.content)
        self.assertIn("第二行", result.content)

    async def test_offset_and_limit(self) -> None:
        self.write("a.txt", "\n".join(f"line{index}" for index in range(1, 11)))
        tool = make_read_tool(self.cwd)

        result = await tool.handler({"path": "a.txt", "offset": 3, "limit": 2}, self.context)

        self.assertIn("line3", result.content)
        self.assertIn("line4", result.content)
        self.assertNotIn("line5", result.content)
        self.assertIn("显示第 3-4 行", result.content)

    async def test_missing_file_is_a_structured_error(self) -> None:
        result = await make_read_tool(self.cwd).handler({"path": "nope.txt"}, self.context)

        self.assertTrue(result.is_error)
        self.assertIn("文件不存在", result.content)

    async def test_directory_is_refused(self) -> None:
        (self.cwd / "sub").mkdir()
        result = await make_read_tool(self.cwd).handler({"path": "sub"}, self.context)

        self.assertTrue(result.is_error)
        self.assertIn("是目录", result.content)

    async def test_missing_path_argument(self) -> None:
        result = await make_read_tool(self.cwd).handler({}, self.context)
        self.assertTrue(result.is_error)

    async def test_safety_limit_truncates(self) -> None:
        """`max_output` 现在是"别把内存读爆"的安全上限,上下文预算归 spill 策略管。"""
        self.write("big.txt", "x" * 500)
        tool = make_read_tool(self.cwd, max_output=100)

        result = await tool.handler({"path": "big.txt"}, self.context)

        self.assertIn("超过安全上限", result.content)
        self.assertLess(len(result.content), 200)

    async def test_path_escape_is_refused(self) -> None:
        outside = self.cwd.parent / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        try:
            result = await make_read_tool(self.cwd).handler(
                {"path": "../outside.txt"}, self.context
            )
            self.assertTrue(result.is_error)
            self.assertIn("路径越界", result.content)
        finally:
            outside.unlink(missing_ok=True)

    async def test_allow_outside_opens_it_up(self) -> None:
        outside = self.cwd.parent / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        try:
            result = await make_read_tool(self.cwd, allow_outside=True).handler(
                {"path": "../outside.txt"}, self.context
            )
            self.assertFalse(result.is_error)
            self.assertIn("secret", result.content)
        finally:
            outside.unlink(missing_ok=True)


class WriteFileTests(_ToolTests):
    async def test_creates_file_and_parents(self) -> None:
        tool = make_write_tool(self.cwd)

        result = await tool.handler(
            {"path": "deep/nested/a.txt", "content": "内容"}, self.context
        )

        self.assertFalse(result.is_error)
        self.assertIn("已创建", result.content)
        self.assertEqual(
            (self.cwd / "deep" / "nested" / "a.txt").read_text(encoding="utf-8"), "内容"
        )

    async def test_overwrite_reports_before_and_after_size(self) -> None:
        self.write("a.txt", "12345")
        tool = make_write_tool(self.cwd)

        result = await tool.handler({"path": "a.txt", "content": "abc"}, self.context)

        self.assertIn("已覆盖", result.content)
        self.assertIn("5 → 3", result.content)

    async def test_missing_content_argument(self) -> None:
        result = await make_write_tool(self.cwd).handler({"path": "a.txt"}, self.context)
        self.assertTrue(result.is_error)
        self.assertIn("content", result.content)

    async def test_writing_to_a_directory_is_refused(self) -> None:
        (self.cwd / "sub").mkdir()
        result = await make_write_tool(self.cwd).handler(
            {"path": "sub", "content": "x"}, self.context
        )
        self.assertTrue(result.is_error)

    async def test_path_escape_is_refused(self) -> None:
        result = await make_write_tool(self.cwd).handler(
            {"path": "../evil.txt", "content": "x"}, self.context
        )

        self.assertTrue(result.is_error)
        self.assertIn("路径越界", result.content)
        self.assertFalse((self.cwd.parent / "evil.txt").exists())


class RegistrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    async def test_default_tree_exposes_the_builtin_tools(self) -> None:
        ctx = build_test_context(scripted_plugin([{"text": "ok"}]), self.cwd)

        names = [schema.name for schema in ctx.tools.schemas()]

        self.assertEqual(names, ["read", "write", "edit", "glob", "grep", "read_image", "pwsh", "job_list", "job_output", "job_kill", "web_search", "web_fetch", "ask_user_question", "present", "task"])

    async def test_agent_can_write_then_read_back_in_one_turn(self) -> None:
        """模型视角的完整闭环:写一个文件,再读回来。"""
        ctx = build_test_context(
            scripted_plugin(
                [
                    {
                        "tool_calls": [
                            {
                                "name": "write",
                                "arguments": {"file_path": "note.md", "content": "# 标题"},
                            }
                        ]
                    },
                    {
                        "tool_calls": [
                            {"name": "read", "arguments": {"file_path": "note.md"}}
                        ]
                    },
                    {"text": "写好了,内容确认无误"},
                ]
            ),
            self.cwd,
            approval="allow",
        )

        result = await ctx.agents.create(ctx.sessions.create()).run("写个文件再读回来")

        self.assertEqual(result.stopped, "final")
        self.assertEqual(result.steps, 3)
        self.assertIn("# 标题", result.session.events_of("tool/result")[1].data["content"])


if __name__ == "__main__":
    unittest.main()
