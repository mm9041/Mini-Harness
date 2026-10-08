"""工具结果外溢测试。

要证明的是一件事:**"太大"不该等于"丢内容"**。

- 存储层:落盘、文件名不可预测、按会话分目录、过期清理;
- 策略层:超预算才动、预览里有 locator 与**检索指引**、`is_error` 原样保留、
  存不下时退化成硬截断(而不是把 20 万字符原样交回去);
- 接线层:挂在 `tools/post-execute` 上,所以**任何**工具都受管 ——
  包括之后新加的工具,一行都不用改;
- 闭环:模型能凭 locator 用 `read_file` 把全文读回来(外溢目录是只读白名单)。
"""

from __future__ import annotations

import time
import unittest
from pathlib import Path

from mini_harness.adapters.mock import scripted_plugin
from mini_harness.llm import ToolCall
from mini_harness.spill import (
    LocalSpillStore,
    SpillPolicy,
    SpillRecord,
    default_root,
)
from mini_harness.tools import ToolCallContext, ToolResult

from . import support
from .support import build_context_with as build_test_context

MARKER = "深层细节-DEEP-DETAIL"


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = support.make_temp_dir(prefix="mini-harness-spill-")

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)

    def test_save_round_trips_and_reports_sizes(self) -> None:
        store = LocalSpillStore(self.root)
        text = "内容" * 100

        record = store.save(text, session_id="s1", hint="pwsh")

        self.assertTrue(Path(record.locator).is_file())
        self.assertEqual(record.chars, len(text))
        self.assertEqual(record.bytes, len(text.encode("utf-8")))
        self.assertEqual(Path(record.locator).read_text(encoding="utf-8"), text)

    def test_files_group_by_session_under_stable_dirs(self) -> None:
        store = LocalSpillStore(self.root)
        first = store.save("a", session_id="session-a")
        second = store.save("b", session_id="session-b")

        self.assertEqual(Path(first.locator).parent, self.root / "session-a")
        self.assertEqual(Path(second.locator).parent, self.root / "session-b")

    def test_names_are_not_predictable(self) -> None:
        """文件名必须不可猜 —— 可猜就能被预先种下符号链接把输出重定向走。"""
        store = LocalSpillStore(self.root)
        names = {Path(store.save("x", session_id="s").locator).name for _ in range(5)}
        self.assertEqual(len(names), 5)  # 同名会覆盖,这里必须每次都不同

    def test_hint_is_slugged_into_the_name(self) -> None:
        store = LocalSpillStore(self.root)
        record = store.save("x", session_id="s", hint="pwsh / 危险:字符")
        self.assertNotIn("/", Path(record.locator).name)
        self.assertIn("pwsh", Path(record.locator).name)

    def test_cleanup_removes_expired_sessions_only(self) -> None:
        store = LocalSpillStore(self.root, retention_days=7)
        fresh = store.save("新", session_id="fresh")
        stale = store.save("旧", session_id="stale")

        old = time.time() - 30 * 86400
        stale_dir = Path(stale.locator).parent
        for child in stale_dir.iterdir():
            import os

            os.utime(child, (old, old))
        import os

        os.utime(stale_dir, (old, old))

        removed = store.cleanup()

        self.assertIn("stale", removed)
        self.assertTrue(Path(fresh.locator).is_file())
        self.assertFalse(stale_dir.exists())

    def test_cleanup_can_be_disabled(self) -> None:
        store = LocalSpillStore(self.root, retention_days=0)
        record = store.save("x", session_id="s")
        import os

        os.utime(Path(record.locator).parent, (0, 0))
        self.assertEqual(store.cleanup(), [])
        self.assertTrue(Path(record.locator).is_file())

    def test_default_root_is_outside_the_workspace(self) -> None:
        self.assertEqual(default_root(), Path.home() / ".mini-harness" / "spill")


class PolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = support.make_temp_dir(prefix="mini-harness-spill-")
        self.store = LocalSpillStore(self.root)
        self.policy = SpillPolicy(max_inline_chars=200, head_chars=40, tail_chars=30)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)

    def test_small_result_passes_through_untouched(self) -> None:
        original = ToolResult("小结果")
        result, record = self.policy.apply(
            original, store=self.store, session_id="s"
        )
        self.assertIs(result, original)
        self.assertIsNone(record)

    def test_large_result_becomes_a_preview_with_retrieval_guidance(self) -> None:
        text = "HEAD" + MARKER * 100 + "TAIL"
        result, record = self.policy.apply(
            ToolResult(text), store=self.store, session_id="s", hint="pwsh"
        )

        self.assertIsNotNone(record)
        self.assertLess(len(result.content), len(text))
        self.assertIn("HEAD", result.content)
        self.assertIn("TAIL", result.content)
        self.assertIn("中间省略", result.content)
        # 指引必须让模型知道去哪取细节,否则它会拿残缺内容硬答
        self.assertIn(record.locator, result.content)
        self.assertIn("read", result.content)
        # 全文一个字都没丢
        self.assertEqual(Path(record.locator).read_text(encoding="utf-8"), text)

    def test_is_error_is_preserved(self) -> None:
        result, _ = self.policy.apply(
            ToolResult("x" * 500, is_error=True), store=self.store, session_id="s"
        )
        self.assertTrue(result.is_error)

    def test_storage_failure_falls_back_to_a_bounded_truncation(self) -> None:
        class BrokenStore:
            def save(self, text, *, session_id, hint=""):  # noqa: ANN001, ARG002
                raise OSError("磁盘满了")

        text = "A" * 1000
        result, record = self.policy.apply(
            ToolResult(text), store=BrokenStore(), session_id="s"
        )

        self.assertIsNone(record)
        self.assertLess(len(result.content), len(text))
        self.assertIn("不可恢复", result.content)  # 明说拿不回来,而不是假装没事

    def test_policy_can_be_disabled_by_a_huge_budget(self) -> None:
        policy = SpillPolicy(max_inline_chars=10**9)
        result, record = policy.apply(
            ToolResult("x" * 5000), store=self.store, session_id="s"
        )
        self.assertIsNone(record)
        self.assertEqual(result.content, "x" * 5000)

    def test_preview_always_fits_the_budget(self) -> None:
        """预算比"头+尾"还小时,头尾要按比例缩 —— 否则预览比原文还长,外溢就没意义了。"""
        policy = SpillPolicy(max_inline_chars=200, head_chars=5000, tail_chars=5000)
        text = "x" * 40000

        result, record = policy.apply(ToolResult(text), store=self.store, session_id="s")

        self.assertIsNotNone(record)
        # 预算 200 加上省略标记与指引文字,给一点余量
        self.assertLess(len(result.content), 200 + 400)
        self.assertLess(len(result.content), len(text))


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    """真正走一遍工具流水线:外溢挂在 tools/post-execute 上,工具本身不用改。"""

    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()
        self.big = "头" + MARKER * 4000 + "尾"  # 远超预算
        (self.cwd / "big.txt").write_text(self.big, encoding="utf-8")

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    async def test_oversized_tool_result_is_spilled_inside_the_pipeline(self) -> None:
        ctx = build_test_context(
            scripted_plugin(
                [
                    {"tool_calls": [{"name": "read", "arguments": {"file_path": "big.txt"}}]},
                    {"text": "读完了"},
                ]
            ),
            self.cwd,
            spill_max_inline_chars=500,
        )
        session = ctx.sessions.create()

        result = await ctx.agents.create(session).run("读这个大文件")

        content = result.session.events_of("tool/result")[0].data["content"]
        self.assertLess(len(content), 4000, "落进日志的必须是预览,不是全文")
        self.assertIn("已保存到", content)

        locator = next(
            token for token in content.replace("]", " ").split() if token.endswith(".txt")
        )
        self.assertEqual(Path(locator).read_text(encoding="utf-8"), self.big)

        # 闭环:模型能凭 locator 用 read_file 把全文读回来(外溢目录是只读白名单)
        back = await ctx.tools.execute(
            ToolCall(id="c2", name="read", arguments={"file_path": locator}),
            cwd=self.cwd,
        )
        self.assertFalse(back.is_error, back.content)
        self.assertIn(MARKER, back.content)

    async def test_spill_disabled_keeps_everything_inline(self) -> None:
        ctx = build_test_context(
            scripted_plugin(
                [
                    {"tool_calls": [{"name": "read", "arguments": {"file_path": "big.txt"}}]},
                    {"text": "读完了"},
                ]
            ),
            self.cwd,
            spill=False,
        )

        result = await ctx.agents.create(ctx.sessions.create()).run("读这个大文件")

        content = result.session.events_of("tool/result")[0].data["content"]
        self.assertIn(MARKER, content)
        self.assertNotIn("已保存到", content)
        self.assertEqual(result.session.events_of("spill/saved"), [])

    async def test_write_file_does_not_get_write_access_to_the_spill_dir(self) -> None:
        """读放宽、写锁死:外溢目录只对 read_file 开放。

        注意外溢目录必须**在工作区之外**才能测出这件事 ——
        测试脚手架默认把 spill_root 放在工作区里(为了好清理),这里显式挪出去。
        """
        outside = support.make_temp_dir(prefix="mini-harness-spill-outside-")
        try:
            # approval=allow:否则审批闸门先拦下来,就测不到路径检查了
            ctx = build_test_context(
                scripted_plugin([{"text": "x"}]),
                self.cwd,
                approval="allow",
                spill_root=str(outside),
            )
            record = ctx.spillStore.save("内容", session_id="s1")

            written = await ctx.tools.execute(
                ToolCall(
                    id="c1",
                    name="write",
                    arguments={"file_path": record.locator, "content": "篡改"},
                ),
                cwd=self.cwd,
            )

            self.assertTrue(written.is_error)
            self.assertIn("路径越界", written.content)
            self.assertEqual(Path(record.locator).read_text(encoding="utf-8"), "内容")

            # 但读是允许的 —— 这正是闭环的另一半
            read_back = await ctx.tools.execute(
                ToolCall(id="c2", name="read", arguments={"file_path": record.locator}),
                cwd=self.cwd,
            )
            self.assertFalse(read_back.is_error, read_back.content)
            self.assertIn("内容", read_back.content)
        finally:
            await support.remove_tree(outside)


if __name__ == "__main__":
    unittest.main()
