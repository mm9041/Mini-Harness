import unittest
from unittest.mock import patch

from mini_harness.adapters.mock import scripted_plugin
from mini_harness.kernel import Context, MODE_WATERFALL
from mini_harness.llm import ToolCall
from mini_harness.tools import Tool, ToolsService
from .support import build_context_with, make_temp_dir, remove_tree


class UnknownToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = make_temp_dir()

    async def asyncTearDown(self):
        await remove_tree(self.root)

    async def test_unknown_tool_never_enters_permission_pipeline(self):
        for preset, approval in ((None,'ask'), ('workspace-write','ask'),
                                 ('workspace-write','allow'), ('danger-full-access','allow')):
            with self.subTest(preset=preset, approval=approval):
                ctx = build_context_with(scripted_plugin([]), self.root,
                                         permission_preset=preset, approval=approval)
                try:
                    session = ctx.sessions.create()
                    with patch.object(ctx, 'waterfall', wraps=ctx.waterfall) as dispatch:
                        result = await ctx.tools.execute(ToolCall('missing','does_not_exist'), session, self.root)
                    self.assertTrue(result.is_error)
                    self.assertEqual(result.content, "未知工具 'does_not_exist';可用工具: " + str(sorted(tool.name for tool in ctx.tools.schemas())))
                    dispatch.assert_not_awaited()
                    self.assertEqual(ctx.approval.history, [])
                    self.assertNotIn('danger-full-access', result.content)
                finally:
                    ctx.dispose()


class ToolRegistrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ctx = Context()
        self.addCleanup(self.ctx.dispose)
        self.tools = ToolsService(self.ctx)

    async def test_stale_disposer_cannot_remove_replacement(self):
        original = Tool('probe','first',handler=lambda args,ctx:'first')
        old = self.tools.register(original)
        old()
        current = self.tools.register(Tool('probe','second',handler=lambda args,ctx:'second'))
        old()
        self.assertTrue(self.tools.has('probe'))
        self.assertEqual((await self.tools.execute(ToolCall('call','probe'))).content, 'second')
        current()
        current()
        self.assertFalse(self.tools.has('probe'))

    async def test_old_disposer_is_inert_when_same_object_is_registered_again(self):
        tool = Tool('probe','test',handler=lambda args,ctx:'ok')
        old = self.tools.register(tool)
        old()
        self.tools.register(tool)
        old()
        self.assertTrue(self.tools.has('probe'))

    async def test_replacement_during_pre_execute_does_not_run(self):
        executed = []
        old = self.tools.register(Tool('probe','first',handler=lambda a,c:executed.append('first')))
        async def replace(call, context, nxt):
            old()
            self.tools.register(Tool('probe','second',handler=lambda a,c:executed.append('second')))
            return await nxt()
        self.ctx.on('tools/pre-execute', replace, MODE_WATERFALL)
        result = await self.tools.execute(ToolCall('call','probe'))
        self.assertTrue(result.is_error)
        self.assertIn('发生变化', result.content)
        self.assertEqual(executed, [])

    async def test_registered_tool_with_unknown_permission_still_reaches_gate(self):
        from mini_harness.approval import plugin
        from mini_harness.kernel import mount
        from mini_harness.tools import plugin as tools_plugin
        mount(self.ctx, [tools_plugin(), plugin(mode='deny')])
        executed = []
        self.ctx.tools.register(Tool('custom','undeclared permission',handler=lambda a,c:executed.append(True)))
        result = await self.ctx.tools.execute(ToolCall('call','custom'))
        self.assertTrue(result.is_error)
        self.assertIn('审批策略拒绝', result.content)
        self.assertEqual(executed, [])
