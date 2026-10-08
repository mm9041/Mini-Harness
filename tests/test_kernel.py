"""内核测试:服务、依赖推导、可撤销效果、三种事件调度。

用标准库 unittest 就能跑,不需要 pytest 或任何第三方依赖:

    python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from mini_harness.kernel import (
    MODE_EMIT,
    MODE_SERIAL,
    MODE_WATERFALL,
    Context,
    MountError,
    Plugin,
    ServiceNotFound,
    mount,
)


class ServiceTests(unittest.TestCase):
    def test_provide_and_get(self) -> None:
        ctx = Context()
        ctx.provide("llm", "adapter")
        self.assertTrue(ctx.has("llm"))
        self.assertEqual(ctx.llm, "adapter")
        self.assertEqual(ctx.get("llm"), "adapter")

    def test_missing_service_raises_with_hint(self) -> None:
        ctx = Context()
        ctx.provide("llm", "adapter")
        with self.assertRaises(ServiceNotFound) as caught:
            ctx.get("tools")
        self.assertIn("llm", str(caught.exception))

    def test_duplicate_provide_rejected(self) -> None:
        ctx = Context()
        ctx.provide("llm", "a")
        with self.assertRaises(MountError):
            ctx.provide("llm", "b")

    def test_scope_sees_parent_and_can_override(self) -> None:
        parent = Context(label="root")
        parent.provide("tools", "parent-tools")
        child = parent.scope("agent-1")

        self.assertEqual(child.tools, "parent-tools")  # 沿父链查找
        child.provide("tools", "child-tools")  # 在子作用域覆盖
        self.assertEqual(child.tools, "child-tools")
        self.assertEqual(parent.tools, "parent-tools")  # 父级不受影响


class MountTests(unittest.TestCase):
    def test_build_context_rolls_back_failed_mount(self):
        from mini_harness.app import build_context, HarnessConfig
        seen, released = [], []
        def first(ctx):
            seen.append(ctx)
            ctx.provide('x', 1)
            ctx.effect(lambda: released.append('closed'))
        plugins = [Plugin('first', first), Plugin('duplicate', lambda ctx: ctx.provide('x', 2))]
        with patch('mini_harness.app.build_plugins', return_value=plugins):
            with self.assertRaises(MountError):
                build_context(HarnessConfig())
        self.assertEqual(seen[0].available_keys(), set())
        self.assertEqual(released, ['closed'])

    def test_inject_derives_load_order(self) -> None:
        ctx = Context()
        order: list[str] = []

        def make(name: str, *inject: str) -> Plugin:
            def apply(c: Context) -> None:
                order.append(name)
                c.provide(name, name)

            return Plugin(name=name, apply=apply, inject=inject)

        # 故意把声明顺序倒过来:下游在前,上游在后。
        mount(ctx, [make("c", "b"), make("b", "a"), make("a")])

        self.assertEqual(order, ["a", "b", "c"])

    def test_unsatisfiable_dependency_is_reported(self) -> None:
        ctx = Context()

        def apply(c: Context) -> None:  # pragma: no cover - 不该被调用
            raise AssertionError("依赖未满足的插件不应被装载")

        with self.assertRaises(MountError) as caught:
            mount(ctx, [Plugin(name="broken", apply=apply, inject=("nope",))])
        self.assertIn("broken", str(caught.exception))


class EffectTests(unittest.TestCase):
    def test_dispose_unwinds_in_reverse_order(self) -> None:
        ctx = Context()
        unwound: list[str] = []
        ctx.effect(lambda: unwound.append("first"))
        ctx.effect(lambda: unwound.append("second"))

        ctx.dispose()

        self.assertEqual(unwound, ["second", "first"])
        self.assertFalse(ctx.has("anything"))

    def test_manual_undo_is_idempotent(self) -> None:
        ctx = Context()
        calls: list[int] = []
        undo = ctx.effect(lambda: calls.append(1))

        undo()
        undo()
        ctx.dispose()

        self.assertEqual(calls, [1])

    def test_registration_after_dispose_rolls_back_immediately(self) -> None:
        ctx = Context()
        ctx.dispose()
        calls: list[int] = []
        ctx.effect(lambda: calls.append(1))
        self.assertEqual(calls, [1])


class EventTests(unittest.IsolatedAsyncioTestCase):
    async def test_terminal_observes_replaced_arguments_without_changing_default_semantics(self):
        ctx = Context()
        async def first(value, nxt):
            return await nxt(value + '-first')
        async def second(value, nxt):
            return await nxt(value + '-second')
        ctx.on('rewrite', first, MODE_WATERFALL)
        ctx.on('rewrite', second, MODE_WATERFALL)
        self.assertEqual(await ctx.waterfall('rewrite', 'input', default='original'), 'original')
        self.assertEqual(await ctx.waterfall('rewrite', 'input', terminal=lambda value: value), 'input-first-second')

    async def test_registration_conflicts_fail_before_dispatch(self):
        root = Context()
        child = root.scope('child')
        seen = []
        undo = child.on('demo', lambda: seen.append('called'), MODE_EMIT)
        with self.assertRaises(MountError):
            root.on('demo', lambda: None, MODE_SERIAL)
        with self.assertRaises(MountError):
            child.on('demo', lambda: None, MODE_WATERFALL)
        self.assertEqual(seen, [])
        undo()
        root.on('demo', lambda: 42, MODE_SERIAL)
        self.assertEqual(await child.serial('demo'), 42)

    async def test_modes_follow_live_registrations_and_are_tree_local(self):
        root = Context()
        child = root.scope('child')
        undo = root.on('demo', lambda: None, MODE_EMIT)
        with self.assertRaises(MountError):
            child.on('demo', lambda: None, MODE_SERIAL)
        child.on('demo', lambda: None, MODE_EMIT)
        undo()
        with self.assertRaises(MountError):
            root.on('demo', lambda: None, MODE_SERIAL)
        other = Context()
        other.on('demo', lambda: None, MODE_SERIAL)
        child.dispose()
        root.on('demo', lambda: 9, MODE_SERIAL)
        self.assertEqual(await root.serial('demo'), 9)

    async def test_emit_runs_listeners_in_registration_order(self) -> None:
        ctx = Context()
        seen: list[str] = []

        async def first(*_args: object) -> None:
            seen.append("first")

        def second(*_args: object) -> None:
            seen.append("second")

        ctx.on("demo", first, mode=MODE_EMIT)
        ctx.on("demo", second, mode=MODE_EMIT)

        await ctx.emit("demo")

        self.assertEqual(seen, ["first", "second"])

    async def test_waterfall_wraps_and_short_circuits(self) -> None:
        ctx = Context()

        async def uppercase(text: str, nxt) -> str:
            return (await nxt()).upper()

        def short_circuit(text: str, nxt) -> str:
            return "被拦截"

        ctx.on("render", uppercase, mode=MODE_WATERFALL)
        self.assertEqual(await ctx.waterfall("render", "hi", default="hi"), "HI")

        ctx.on("render", short_circuit, mode=MODE_WATERFALL)
        self.assertEqual(await ctx.waterfall("render", "hi", default="hi"), "被拦截")

    async def test_waterfall_can_rewrite_in_place_and_delegate(self) -> None:
        ctx = Context()
        payload = {"value": 1}

        async def bump(data: dict, nxt) -> dict:
            data["value"] += 1
            return await nxt()

        ctx.on("mutate", bump, mode=MODE_WATERFALL)
        result = await ctx.waterfall("mutate", payload, default=payload)

        self.assertIs(result, payload)
        self.assertEqual(result["value"], 2)

    async def test_serial_returns_last_result(self) -> None:
        ctx = Context()
        ctx.on("pipeline", lambda: 1, mode=MODE_SERIAL)
        ctx.on("pipeline", lambda: 2, mode=MODE_SERIAL)

        self.assertEqual(await ctx.serial("pipeline"), 2)

    async def test_mode_mismatch_is_a_programming_error(self) -> None:
        ctx = Context()
        ctx.on("demo", lambda: None, mode=MODE_SERIAL)

        with self.assertRaises(RuntimeError):
            await ctx.emit("demo")

    async def test_child_scope_listener_runs_before_parent(self) -> None:
        parent = Context(label="root")
        child = parent.scope("child")
        order: list[str] = []

        parent.on("demo", lambda: order.append("parent"), mode=MODE_EMIT)
        child.on("demo", lambda: order.append("child"), mode=MODE_EMIT)

        await child.emit("demo")

        self.assertEqual(order, ["child", "parent"])

    async def test_listener_disposed_with_context(self) -> None:
        ctx = Context()
        seen: list[int] = []
        ctx.on("demo", lambda: seen.append(1), mode=MODE_EMIT)

        ctx.dispose()
        await ctx.emit("demo")

        self.assertEqual(seen, [])


if __name__ == "__main__":
    unittest.main()
