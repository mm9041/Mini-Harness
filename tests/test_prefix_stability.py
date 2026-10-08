"""前缀稳定性回归测试。

不变量：厂商、模型和思考设置固定时，第 i 条消息的 wire 形态只取决于自身内容，
不应取决于它在列表中的位置，也不应取决于会话走到了哪一步。
因此"在尾部追加新消息"不得改写任何一条已经发送过的消息 ——
这是服务商前缀缓存（KV cache / prompt cache）能够命中的前提。

为什么需要这条断言：没有它，"某条旧消息的序列化形态发生改变"这类缺陷
可以在全部测试通过、日志追加式不变的情况下长期存在，而且在客户端完全不可见。

历史工具消息保留已记录的 reasoning；关闭思考或切换到不使用该字段的厂商时仍按
对应协议处理。前缀稳定只消除客户端的改写，不保证服务商实际命中或某个计费折扣。
"""

import json
from pathlib import Path
import tempfile
import unittest

from mini_harness.adapters.openai_compat import OpenAICompatAdapter
from mini_harness.llm import GenerateRequest, GenerateResult, Message, ToolCall


def wire(adapter, messages, system=''):
    """取一次请求真正会上线的 messages 数组。"""
    request = GenerateRequest(system=system, messages=messages)
    return adapter._build_payload(request, stream=True)['messages']


def common_prefix(a, b):
    """两条 wire 消息列表的最长公共前缀长度。"""
    length = 0
    while length < len(a) and length < len(b):
        if json.dumps(a[length], sort_keys=True, ensure_ascii=False) != \
           json.dumps(b[length], sort_keys=True, ensure_ascii=False):
            break
        length += 1
    return length


class PrefixStabilityTests(unittest.TestCase):
    def make_adapter(self, effort='high', model='deepseek-v4-flash'):
        adapter = OpenAICompatAdapter('test-key', model=model)
        adapter.reasoning_effort = effort
        return adapter

    # ---------------------------------------------------------------- 对照组
    def test_tail_append_within_a_step_loop_keeps_every_earlier_message(self):
        """同一轮内追加 assistant/tool 消息：应当稳定。"""
        adapter = self.make_adapter()
        first = [Message('user', content='task one')]
        before = wire(adapter, first)

        second = first + [
            Message('assistant', tool_calls=[ToolCall('c1', 'read')], reasoning='r1'),
            Message('tool', content='res1', tool_call_id='c1'),
        ]
        after = wire(adapter, second)

        self.assertEqual(after[:len(before)], before)

    # ---------------------------------------------------------------- 回归点
    def test_tail_append_across_a_turn_boundary_keeps_every_earlier_message(self):
        """追加一条 user 消息 = 新的一轮开始。已发送的消息一条都不该变。

        这是消息前缀保持一致的必要条件。若这里失败，说明有代码在用
        "消息在列表中的位置" 决定它的序列化形态。
        """
        adapter = self.make_adapter()
        turn_one = [
            Message('user', content='task one'),
            Message('assistant', tool_calls=[ToolCall('c1', 'read')], reasoning='r1'),
            Message('tool', content='res1', tool_call_id='c1'),
            Message('assistant', content='done one'),
        ]
        before = wire(adapter, turn_one)

        turn_two = turn_one + [Message('user', content='next task')]
        after = wire(adapter, turn_two)

        kept = common_prefix(before, after)
        # 区分两件不同的事：内容真的变了（改写），和从分叉点起丢失前缀复用（作废）。
        rewritten = sum(
            1 for index in range(kept, len(before))
            if json.dumps(before[index], sort_keys=True) != json.dumps(after[index], sort_keys=True)
        )
        detail = ''
        if kept < len(before):
            detail = (
                f'\n  首个分叉点是第 {kept} 条：'
                f'\n    before = {json.dumps(before[kept], ensure_ascii=False)}'
                f'\n    after  = {json.dumps(after[kept], ensure_ascii=False)}'
            )
        self.assertEqual(
            kept, len(before),
            f'尾部追加一条 user 消息后：{len(before)} 条已发送消息里有 {rewritten} 条内容被改写，'
            f'另外 {len(before) - kept} 条（含分叉点自身）丢失前缀复用。{detail}',
        )

    # ---------------------------------------------------------------- 旁证
    def test_reasoning_replay_of_a_message_does_not_depend_on_list_position(self):
        """把同一条 assistant 消息放在不同位置，它的 wire 形态应当一致。"""
        adapter = self.make_adapter()
        assistant = Message('assistant', tool_calls=[ToolCall('c1', 'read')], reasoning='r1')

        alone = wire(adapter, [Message('user', content='u1'), assistant])
        with_tail = wire(adapter, [Message('user', content='u1'), assistant,
                                   Message('tool', content='res', tool_call_id='c1'),
                                   Message('user', content='u2')])

        self.assertEqual(
            json.dumps(alone[1], sort_keys=True),
            json.dumps(with_tail[1], sort_keys=True),
            '同一条 assistant 消息在两个列表里的 wire 形态不同，'
            '说明它的序列化依赖了它在列表中的位置。',
        )

    def test_appending_own_tool_results_does_not_change_the_assistant(self):
        adapter = self.make_adapter()
        messages = [Message('user', content='task'), Message('assistant', reasoning='recorded reasoning',
                    tool_calls=[ToolCall('one','read'), ToolCall('two','grep')])]
        before = wire(adapter, messages)
        self.assertEqual(before[-1]['reasoning_content'], 'recorded reasoning')
        for call_id in ('one', 'two'):
            messages.append(Message('tool', content='result', tool_call_id=call_id))
            self.assertEqual(wire(adapter, messages)[:len(before)], before)

    def test_provider_and_message_gates_are_preserved(self):
        messages = [Message('assistant', reasoning='tool reasoning', tool_calls=[ToolCall('one','read')]),
                    Message('assistant', content='final', reasoning='not tool reasoning'),
                    Message('assistant', tool_calls=[ToolCall('two','read')]),
                    Message('user', content='next task')]
        for model in ('deepseek-v4-flash', 'kimi-k3', 'glm-5.3'):
            adapter = self.make_adapter(model=model)
            for stream in (False, True):
                with self.subTest(model=model, stream=stream):
                    payload = adapter._build_payload(GenerateRequest(system='', messages=messages), stream=stream)
                    self.assertEqual(payload['messages'][0]['reasoning_content'], 'tool reasoning')
                    self.assertNotIn('reasoning_content', payload['messages'][1])
                    self.assertNotIn('reasoning_content', payload['messages'][2])
        # Qwen and generic OpenAI keep their existing protocol; no new field is added.
        for model in ('qwen-plus', 'custom-openai-model'):
            self.assertNotIn('reasoning_content', wire(self.make_adapter(model=model), messages)[0])
        self.assertNotIn('reasoning_content', wire(self.make_adapter(effort='none'), messages)[0])


class SummaryPrefixTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_summary_request_keeps_covered_message_prefix(self):
        from mini_harness.adapters.mock import ScriptedAdapter
        from .support import adapter_plugin_for, build_context_with
        recorder = ScriptedAdapter([GenerateResult(text='summary')])
        with tempfile.TemporaryDirectory() as directory:
            ctx = build_context_with(adapter_plugin_for(recorder), Path(directory))
            try:
                serializer = OpenAICompatAdapter('test-key', model='deepseek-v4-flash')
                serializer.reasoning_effort = 'high'
                messages = [Message('user', content='task'),
                            Message('assistant', reasoning='r1', tool_calls=[ToolCall('one','read')]),
                            Message('tool', content='result1', tool_call_id='one'),
                            Message('assistant', reasoning='r2', tool_calls=[ToolCall('two','grep')]),
                            Message('tool', content='result2', tool_call_id='two')]
                main = serializer._build_payload(GenerateRequest(system=ctx.systemPrompt.render(),
                    messages=messages, tools=ctx.tools.schemas()), stream=True)
                self.assertEqual(await ctx.compaction._summarize(messages), 'summary')
                summary = serializer._build_payload(recorder.requests[-1], stream=False)
                self.assertEqual(summary['messages'][:-1], main['messages'])
                self.assertEqual(summary['tools'], main['tools'])
                self.assertEqual(sum('reasoning_content' in m for m in summary['messages']), 2)
                self.assertEqual(sum('reasoning_content' in m for m in main['messages']), 2)
            finally:
                ctx.dispose()
