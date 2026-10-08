"""配置、``.env`` 与端点解析的测试。

注意:凡是调 ``HarnessConfig.from_env`` 的地方都显式传 ``env_file=None`` 或一个临时
路径 —— 否则它会在当前目录/项目根目录找到真实的 ``.env``,把真实凭据读进测试。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mini_harness.adapters.openai_compat import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    OpenAICompatAdapter,
    parse_completion,
    resolve_chat_endpoint,
)
from mini_harness.app import HarnessConfig
from mini_harness.envfile import discover_env_file, load_env_file, parse_env_text


class EnvFileParseTests(unittest.TestCase):
    def test_compaction_window_policy_environment(self):
        defaults = HarnessConfig.from_env({}, env_file=None)
        self.assertEqual((defaults.max_history_tokens, defaults.keep_recent_messages), (0, 0))
        config = HarnessConfig.from_env({
            "MINI_HARNESS_COMPACTION_THRESHOLD_RATIO": "0.7",
            "MINI_HARNESS_COMPACTION_RETAIN_RATIO": "0.12",
            "MINI_HARNESS_COMPACTION_HEADROOM_TOKENS": "1000",
            "MINI_HARNESS_COMPACTION_RESERVED_COMPLETION_TOKENS": "2000",
        }, env_file=None)
        self.assertEqual(config.compaction_threshold_ratio, 0.7)
        self.assertEqual(config.compaction_retain_ratio, 0.12)
        self.assertEqual(config.compaction_headroom_tokens, 1000)
        self.assertEqual(config.compaction_reserved_completion_tokens, 2000)

    def test_parses_plain_export_quoted_and_comments(self) -> None:
        text = "\n".join(
            [
                "# 整行注释",
                "",
                "PLAIN=value",
                "export EXPORTED=value2",
                'DOUBLE="a b"',
                "SINGLE='c d'",
                "TRAILING=value   # 行尾注释",
                "KEEP_HASH='a # b'",
                "WITH_EQUALS=a=b=c",
                "NOT_A_PAIR",
                "=nokey",
            ]
        )

        values = parse_env_text(text)

        self.assertEqual(values["PLAIN"], "value")
        self.assertEqual(values["EXPORTED"], "value2")
        self.assertEqual(values["DOUBLE"], "a b")
        self.assertEqual(values["SINGLE"], "c d")
        self.assertEqual(values["TRAILING"], "value")
        self.assertEqual(values["KEEP_HASH"], "a # b")  # 加引号则不剥注释
        self.assertEqual(values["WITH_EQUALS"], "a=b=c")
        self.assertNotIn("NOT_A_PAIR", values)
        self.assertNotIn("", values)

    def test_quoted_value_with_trailing_comment(self) -> None:
        """引号 + 行尾注释:注释要剥掉,引号也要剥掉。

        回归:判据曾是"首尾字符相同",而这种写法的尾字符是注释的一部分,
        于是走不进引号分支,结果是注释没了、引号却留着(`"hello"`)。
        """
        values = parse_env_text(
            "\n".join(
                [
                    'DOUBLE="hello" # note',
                    "SINGLE='c d'   # note",
                    'NOSPACE="v" #note',
                    "PLAIN=a#b # c",
                    'KEEP=" # "',
                ]
            )
        )

        self.assertEqual(values["DOUBLE"], "hello")
        self.assertEqual(values["SINGLE"], "c d")
        self.assertEqual(values["NOSPACE"], "v")
        self.assertEqual(values["PLAIN"], "a#b")  # 注释前的 # 不算注释
        self.assertEqual(values["KEEP"], " # ")  # 整段带引号则不剥注释

    def test_load_does_not_override_existing_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("FROM_FILE=a\nBOTH=file\n", encoding="utf-8")
            env = {"BOTH": "process"}

            applied = load_env_file(path, env=env)

            self.assertEqual(env["FROM_FILE"], "a")
            self.assertEqual(env["BOTH"], "process")  # 进程里的值优先
            self.assertEqual(applied, {"FROM_FILE": "a"})  # 只报告真正生效的键

    def test_missing_file_is_not_an_error(self) -> None:
        self.assertEqual(load_env_file(Path("no/such/file.env"), env={}), {})

    def test_discover_prefers_explicit_path_then_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("A=cwd\n", encoding="utf-8")
            explicit = root / "custom.env"
            explicit.write_text("A=explicit\n", encoding="utf-8")

            self.assertEqual(
                discover_env_file(
                    start=root, env={"MINI_HARNESS_ENV_FILE": str(explicit)}
                ),
                explicit,
            )
            self.assertEqual(discover_env_file(start=root, env={}), root / ".env")
            self.assertIsNone(
                discover_env_file(
                    start=root,
                    env={"MINI_HARNESS_ENV_FILE": str(root / "missing.env")},
                )
            )


class EndpointResolutionTests(unittest.TestCase):
    def test_dashscope_and_gateway_base_urls(self) -> None:
        cases = {
            "https://dashscope.aliyuncs.com/compatible-mode/v1": (
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
            ),
            "https://maas.qianwenaiapi.com/compatible-mode/v1": (
                "https://maas.qianwenaiapi.com/compatible-mode/v1/chat/completions"
            ),
            "https://dashscope.aliyuncs.com/compatible-mode/v1/": (  # 尾斜杠
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
            ),
            "https://api.openai.com/v1": "https://api.openai.com/v1/chat/completions",
            "https://api.deepseek.com": "https://api.deepseek.com/v1/chat/completions",
            "http://127.0.0.1:8000": "http://127.0.0.1:8000/v1/chat/completions",
            "https://x.example.com/v2": "https://x.example.com/v2/chat/completions",
            "https://x.example.com/v1/chat/completions": (
                "https://x.example.com/v1/chat/completions"
            ),
        }

        for base, expected in cases.items():
            with self.subTest(base=base):
                self.assertEqual(resolve_chat_endpoint(base), expected)

    def test_empty_base_url_is_rejected(self) -> None:
        from mini_harness.llm import LLMError

        with self.assertRaises(LLMError):
            resolve_chat_endpoint("   ")


class ConfigResolutionTests(unittest.TestCase):
    def test_defaults_are_usable_and_self_consistent(self) -> None:
        """不断言默认端点属于哪一家 —— 那是部署选择。

        真正该守住的不变量是:没给任何配置时,默认值必须**可用**
        (非空、能解析出 /chat/completions 端点),而不是半截 URL 或空串。
        """
        config = HarnessConfig.from_env(env={}, env_file=None)

        self.assertEqual(config.base_url, DEFAULT_BASE_URL)
        self.assertEqual(config.model, DEFAULT_MODEL)
        self.assertTrue(DEFAULT_BASE_URL.startswith("https://"), DEFAULT_BASE_URL)
        self.assertTrue(
            resolve_chat_endpoint(DEFAULT_BASE_URL).endswith("/chat/completions"),
            DEFAULT_BASE_URL,
        )
        self.assertTrue(DEFAULT_MODEL)
        self.assertEqual(config.api_key, "")
        self.assertEqual(config.timeout, 120.0)

    def test_dashscope_vars_win_in_documented_order(self) -> None:
        config = HarnessConfig.from_env(
            env={
                "DASHSCOPE_API_KEY": "dash",
                "DEEPSEEK_API_KEY": "deep",
                "OPENAI_API_KEY": "oa",
                "DASHSCOPE_BASE_URL": "https://a.example.com/v1",
                "DEEPSEEK_BASE_URL": "https://b.example.com/v1",
                "DASHSCOPE_MODEL": "m-dash",
                "MINI_HARNESS_MODEL": "m-mini",
            },
            env_file=None,
        )

        self.assertEqual(config.api_key, "dash")
        self.assertTrue(config.api_key_source.startswith("DASHSCOPE_API_KEY"))
        self.assertEqual(config.base_url, "https://a.example.com/v1")
        self.assertEqual(config.model, "m-dash")

    def test_deepseek_only_setup_still_works(self) -> None:
        config = HarnessConfig.from_env(
            env={
                "DEEPSEEK_API_KEY": "deep",
                "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
            },
            env_file=None,
        )

        self.assertEqual(config.api_key, "deep")
        self.assertEqual(config.base_url, "https://api.deepseek.com")

    def test_cli_overrides_win_over_env(self) -> None:
        config = HarnessConfig.from_env(
            env={"DASHSCOPE_API_KEY": "env-key", "DASHSCOPE_MODEL": "env-model"},
            env_file=None,
            api_key="cli-key",
            model="cli-model",
        )

        self.assertEqual(config.api_key, "cli-key")
        self.assertEqual(config.model, "cli-model")
        self.assertIn("命令行", config.api_key_source)

    def test_env_file_is_read_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "DASHSCOPE_API_KEY=from-file\nDASHSCOPE_MODEL=file-model\n",
                encoding="utf-8",
            )

            config = HarnessConfig.from_env(env={}, env_file=path)

            self.assertEqual(config.api_key, "from-file")
            self.assertEqual(config.model, "file-model")
            self.assertEqual(config.api_key_source, "DASHSCOPE_API_KEY (.env)")
            self.assertEqual(config.env_file, str(path))

    def test_process_env_beats_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("DASHSCOPE_API_KEY=from-file\n", encoding="utf-8")
            env = {"DASHSCOPE_API_KEY": "from-process"}

            config = HarnessConfig.from_env(env=env, env_file=path)

            self.assertEqual(config.api_key, "from-process")
            self.assertEqual(config.api_key_source, "DASHSCOPE_API_KEY (环境变量)")

    def test_numeric_settings_from_env(self) -> None:
        config = HarnessConfig.from_env(
            env={
                "MINI_HARNESS_TIMEOUT": "30",
                "MINI_HARNESS_MAX_STEPS": "3",
                "MINI_HARNESS_SHELL_TIMEOUT": "15",
            },
            env_file=None,
        )

        self.assertEqual(config.timeout, 30.0)
        self.assertEqual(config.max_steps, 3)
        self.assertEqual(config.shell_timeout, 15.0)


class SystemPromptTests(unittest.TestCase):
    """人格与运行时上下文 —— 重点是**身份不能留白**。

    留白的代价很具体:persona 只说 "a helpful software engineer assistant" 时,
    模型被问"介绍一下你自己"会**现编一个产品名**。实测同一个请求连发 5 次给出 5 个身份
    (无名的"AI 编程助手"、"小舟"、冒充 "Claude"……)。dsh 用
    `includeHarnessIdentity` 开关控制这件事,这里改成默认把身份锚住。
    """

    def _system_prompt(self, **overrides) -> str:
        from mini_harness.app import HarnessConfig, build_context

        config = HarnessConfig.from_env(env={}, env_file=None, offline=True, **overrides)
        return build_context(config).systemPrompt.render()

    def test_default_persona_anchors_the_identity(self) -> None:
        from mini_harness.app import DEFAULT_PERSONA

        self.assertIn("mini-harness", DEFAULT_PERSONA)
        self.assertIn("never claim", DEFAULT_PERSONA)
        self.assertIn("mini-harness", self._system_prompt())

    def test_runtime_context_carries_the_model_name(self) -> None:
        prompt = self._system_prompt(model="some-model-xyz")
        self.assertIn("模型: some-model-xyz", prompt)

    def test_persona_can_be_overridden(self) -> None:
        prompt = self._system_prompt(persona="You are a terse shell assistant.")
        self.assertIn("You are a terse shell assistant.", prompt)
        self.assertNotIn("never claim", prompt)


class AdapterConstructionTests(unittest.TestCase):
    def test_missing_key_reports_the_right_variable(self) -> None:
        from mini_harness.llm import LLMError

        with self.assertRaises(LLMError) as caught:
            OpenAICompatAdapter(api_key="")
        self.assertIn("DASHSCOPE_API_KEY", str(caught.exception))

    def test_endpoint_uses_resolver(self) -> None:
        adapter = OpenAICompatAdapter(
            api_key="k", base_url="https://maas.example.com/compatible-mode/v1"
        )
        self.assertEqual(
            adapter.endpoint,
            "https://maas.example.com/compatible-mode/v1/chat/completions",
        )

    def test_reasoning_content_is_parsed_and_empty_text_normalized(self) -> None:
        payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "", "reasoning_content": "先想一下……"},
                }
            ]
        }

        result = parse_completion(payload)

        self.assertIsNone(result.text)
        self.assertEqual(result.reasoning, "先想一下……")


if __name__ == "__main__":
    unittest.main()
