"""OpenAI 兼容适配器。

**这是唯一的网络出口**:一个名字 + ``generate``(必需)+ ``stream``(可选),满足
``LLMAdapter`` 协议。默认指向阿里云百炼(DashScope)的 OpenAI 兼容模式::

    base_url = https://dashscope.aliyuncs.com/compatible-mode/v1
    端点     = <base_url>/chat/completions

dsh 里对应的是 ``@deepseek-ai/dsh-llm-deepseek-api-key`` 那一行插件。之所以能一家
适配器通吃,是因为 DashScope 兼容模式、DeepSeek、OpenAI、vLLM / Ollama 的
``/chat/completions`` 说的是同一套协议 —— 这也是"接缝"最实用的地方。

刻意只用标准库:非流式用 ``urllib`` + ``asyncio.to_thread``;流式(SSE)在子线程里
逐行读,再用 ``loop.call_soon_threadsafe`` 把增量投进 asyncio 队列,变成 async 生成器。
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any, AsyncIterator, Iterator

from ..providers import reasoning_options
from ..kernel import Context, Plugin
from ..llm import (
    GenerateRequest,
    GenerateResult,
    LLMError,
    StreamEvent,
    ToolCall,
)

__all__ = [
    "OpenAICompatAdapter",
    "StreamAccumulator",
    "parse_completion",
    "plugin",
    "resolve_chat_endpoint",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "CHAT_PATH",
]

#: DashScope(阿里云百炼)OpenAI 兼容模式端点。
DEFAULT_BASE_URL = "https://maas.qianwenaiapi.com/compatible-mode/v1"
#: 官方 DashScope 的模型名。自建网关/代理常用带版本号的名字(如 qwen3.7-plus),
#: 用 DASHSCOPE_MODEL 或 --model 覆盖即可。
DEFAULT_MODEL = "qwen-plus"

CHAT_PATH = "/chat/completions"


def resolve_chat_endpoint(base_url: str) -> str:
    """把 ``base_url`` 解析成真正要 POST 的端点,容忍几种常见写法。

    ====================================================  ==============================================
    你写的 base_url                                       解析结果
    ====================================================  ==============================================
    ``https://dashscope.aliyuncs.com/compatible-mode/v1``  ``.../compatible-mode/v1/chat/completions``
    ``https://maas.example.com/compatible-mode/v1``        ``.../compatible-mode/v1/chat/completions``
    ``https://api.openai.com/v1``                          ``.../v1/chat/completions``
    ``https://api.deepseek.com``                           ``.../v1/chat/completions``
    ``http://127.0.0.1:8000``                              ``.../v1/chat/completions``
    ``https://x/v1/chat/completions``(已写全)            原样使用
    ====================================================  ==============================================

    规则:已经以 ``/chat/completions`` 结尾就原样用;末段是形如 ``v1`` / ``v2`` 的版本段
    就直接补路径;否则补一个 ``/v1`` 再补路径。
    """
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise LLMError("base_url 为空:请设置 DASHSCOPE_BASE_URL")
    if base.endswith(CHAT_PATH):
        return base
    tail = base.rsplit("/", 1)[-1]
    if len(tail) > 1 and tail[0] == "v" and tail[1:].isdigit():
        return base + CHAT_PATH
    return base + "/v1" + CHAT_PATH


def _try_parse_arguments(raw: str) -> dict[str, Any] | None:
    """参数串能解析成 JSON 对象就返回它,否则 ``None``。"""
    if not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return {"value": parsed} if not isinstance(parsed, dict) else parsed


def _wrap_transport_exception(
    exc: BaseException, endpoint: str, timeout: float
) -> LLMError:
    """把传输层异常翻成 ``LLMError``,并带上**供重试分类用的结构化字段**。

    为什么要在这里就把状态码带上:重试策略要区分"值得重试"(429/5xx/连接抖动)和
    "重试也没用"(400/401/403),靠消息文本去猜是下策 —— 状态码本来就在手里。
    """
    if isinstance(exc, urllib.error.HTTPError):
        detail = exc.read(8192).decode("utf-8", errors="replace")
        fields = {"provider_message": detail}
        try:
            body = json.loads(detail)
            error = body.get("error", body) if isinstance(body, dict) else {}
            if isinstance(error, dict):
                fields = {"provider_" + key: error[key] for key in ("code", "type", "message") if isinstance(error.get(key), str)}
                fields.setdefault("provider_message", "")
        except ValueError:
            pass
        return LLMError(
            f"模型接口返回 HTTP {exc.code}({endpoint}): {detail[:600]}", status=exc.code,
            **fields,
        )
    if isinstance(exc, urllib.error.URLError):
        return LLMError(f"无法连接模型接口 {endpoint}: {exc.reason}", retryable=True)
    if isinstance(exc, TimeoutError):
        return LLMError(f"模型接口超时({timeout:.0f}s): {endpoint}", retryable=True)
    return LLMError(f"模型接口调用失败: {exc!r}", retryable=True)


class _JsonDepth:
    """在字符级跟踪 JSON 的括号深度,跳过字符串内部与转义。

    用途:判断流式送来的工具参数是否**已经闭合**。只看"能不能解析"是不够的 ——
    ``{"a": 1}`` 在 JSON 上合法,但模型可能还要接着写 ``, "b": 2}``。
    深度回到 0 才说明这个对象写完了,此时派发是安全的。
    """

    __slots__ = ("depth", "in_string", "escape")

    def __init__(self) -> None:
        self.depth = 0
        self.in_string = False
        self.escape = False

    def feed(self, text: str) -> None:
        for char in text:
            if self.escape:
                self.escape = False
                continue
            if self.in_string:
                if char == "\\":
                    self.escape = True
                elif char == '"':
                    self.in_string = False
                continue
            if char == '"':
                self.in_string = True
            elif char in "{[":
                self.depth += 1
            elif char in "}]" and self.depth > 0:
                self.depth -= 1

    @property
    def closed(self) -> bool:
        return self.depth == 0 and not self.in_string


class StreamAccumulator:
    """累加 SSE 增量,拼出最终的 ``GenerateResult``,并在参数补全时立刻报出工具调用。

    **它是策略,不是 transport** —— 所以刻意公开:换一套 HTTP 客户端(比如官方 openai
    库)时,这件"把碎片拼成完整调用、并判断参数是否闭合"的事还是同一套逻辑,
    可以原样复用。见 `examples/openai_sdk_adapter.py`。

    单独成类也为了可测:工具调用在流里是**碎片**送达的(名字一次、参数分多次,
    且多个调用按 ``index`` 交错),这块最容易写错,必须能脱离网络用假 chunk 验证。
    """

    def __init__(self) -> None:
        self._text: list[str] = []
        self._reasoning: list[str] = []
        self._tool_slots: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None

    def feed(self, chunk: dict) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        if chunk.get("usage"):
            self.usage = chunk["usage"]

        choices = chunk.get("choices") or []
        if not choices:
            return events
        choice = choices[0]
        if choice.get("finish_reason"):
            self.finish_reason = choice["finish_reason"]
        delta = choice.get("delta") or {}

        content = delta.get("content")
        if content:
            self._text.append(content)
            events.append(StreamEvent("delta", text=content))

        # 思考型模型(Qwen3 / DeepSeek-R1 系)把思维链单独送:留着,但不进模型历史。
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            self._reasoning.append(reasoning)
            events.append(StreamEvent("delta", reasoning=reasoning))

        for raw in delta.get("tool_calls") or []:
            index = int(raw.get("index") or 0)
            slot = self._tool_slots.setdefault(
                index,
                {"id": None, "name": "", "arguments": "", "depth": _JsonDepth(), "dispatched": False},
            )
            if raw.get("id"):
                slot["id"] = raw["id"]
            function = raw.get("function") or {}
            if function.get("name"):
                slot["name"] = (slot["name"] or "") + function["name"]
            fragment = function.get("arguments")
            if fragment:
                slot["arguments"] = (slot["arguments"] or "") + fragment
                slot["depth"].feed(fragment)

            ready = self._maybe_complete(index, slot)
            if ready is not None:
                events.append(ready)
        return events

    def _maybe_complete(self, index: int, slot: dict[str, Any]) -> StreamEvent | None:
        """参数闭合 + id/name 齐备 ⇒ 报一个可立即执行的 ``tool_call`` 帧。"""
        if slot["dispatched"] or not slot["id"] or not slot["name"]:
            return None
        if not slot["depth"].closed:
            return None
        arguments = _try_parse_arguments(slot["arguments"] or "")
        if arguments is None:
            return None
        slot["dispatched"] = True
        return StreamEvent("tool_call", tool_call=self._build_call(index, slot, arguments))

    @staticmethod
    def _build_call(index: int, slot: dict[str, Any], arguments: dict[str, Any]) -> ToolCall:
        raw_arguments = slot["arguments"] or ""
        return ToolCall(
            id=slot["id"] or f"call_{index + 1}",
            name=slot["name"] or "",
            arguments=arguments,
            raw_arguments=raw_arguments or None,
        )

    def finish(self) -> GenerateResult:
        """收尾:无论如何都要给出**完整**的工具调用列表(提前派发过的也在内),
        因为日志里的 ``assistant(tool_calls)`` 必须列出全部调用。"""
        tool_calls: list[ToolCall] = []
        for index in sorted(self._tool_slots):
            slot = self._tool_slots[index]
            parsed = _try_parse_arguments(slot["arguments"] or "")
            tool_calls.append(self._build_call(index, slot, parsed if parsed is not None else {}))
        return GenerateResult(
            text="".join(self._text) or None,
            reasoning="".join(self._reasoning) or None,
            tool_calls=tool_calls,
            finish_reason=self.finish_reason,
            usage=self.usage,
        )


def parse_completion(data: dict) -> GenerateResult:
    """把一次 ``/chat/completions`` 响应(原始 dict)解析成 ``GenerateResult``。

    **和 ``StreamAccumulator`` 一样,这是策略不是 transport** —— 所以公开出来:
    换任何 HTTP 客户端,响应里那点坑(``content`` 可能是分块数组、思考型模型的
    ``reasoning_content``、工具参数的非法 JSON)都还是同一套处理。
    见 `examples/openai_sdk_adapter.py`。
    """
    choices = data.get("choices") or []
    if not choices:
        preview = json.dumps(data, ensure_ascii=False)[:400]
        raise LLMError(f"响应中没有 choices: {preview}")
    choice = choices[0]
    message = choice.get("message") or {}

    tool_calls: list[ToolCall] = []
    for index, raw in enumerate(message.get("tool_calls") or []):
        function = raw.get("function") or {}
        raw_arguments = function.get("arguments")
        try:
            arguments = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError:
            # 模型偶尔会给出非法 JSON:不抛错,保留原始串交给工具层报错。
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        tool_calls.append(
            ToolCall(
                id=raw.get("id") or f"call_{index + 1}",
                name=function.get("name") or "",
                arguments=arguments,
                raw_arguments=raw_arguments,
            )
        )

    text = message.get("content")
    if isinstance(text, list):  # 少数兼容端点把 content 返回成分块结构
        text = "".join(
            part.get("text", "") for part in text if isinstance(part, dict)
        )

    # 思考型模型(Qwen3 / DeepSeek-R1 系)把思维链放在 reasoning_content,
    # 正文可能是空的。它不进模型历史,只在没有正文时兜底。
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(reasoning, list):
        reasoning = "".join(
            part.get("text", "") for part in reasoning if isinstance(part, dict)
        )

    return GenerateResult(
        text=text or None,
        reasoning=reasoning or None,
        tool_calls=tool_calls,
        finish_reason=choice.get("finish_reason"),
        usage=data.get("usage"),
    )


class OpenAICompatAdapter:
    """一个名字 + ``generate`` + ``stream``,满足 LLMAdapter 协议。"""

    name = "openai-compat"

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
    ) -> None:
        if not api_key:
            raise LLMError(
                "缺少 API key:请在 .env 或环境变量里设置 DASHSCOPE_API_KEY"
                "(也认 DEEPSEEK_API_KEY / OPENAI_API_KEY),或用 --mock 离线运行"
            )
        self.api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE_URL).strip()
        self.model = model
        self.model_context_windows: dict[str, int] = {}
        self.reasoning_effort: str | None = None
        self.provider_kind = "auto"
        self.timeout = timeout

    @property
    def endpoint(self) -> str:
        return resolve_chat_endpoint(self.base_url)

    # ------------------------------------------------------------------ 运行时
    def set_model(self, name: str) -> str:
        """换模型,返回旧名字。供 ``/model`` 用。"""
        previous, self.model = self.model, name
        return previous

    async def list_models(self) -> list[str]:
        """问网关有哪些可用模型(``GET /models``)。

        不少兼容网关不支持这个端点,所以调用方必须**兜住异常** —— 拿不到就只显示
        当前模型,不影响干活。
        """
        return await asyncio.to_thread(self._list_models)

    def _list_models(self) -> list[str]:
        # 端点形如 <base>/v1/chat/completions,模型列表在 <base>/v1/models ——
        # 要剥掉 "chat/completions" **两段**,不是一段(剥一段会打到 /v1/chat/models → 404,踩过)。
        url = self.endpoint[: -len(CHAT_PATH)] + "/models"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise _wrap_transport_exception(exc, url, self.timeout) from exc

        items = data.get("data") or []
        for item in items:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            capacity = next((item.get(key) for key in ("context_window", "context_length", "max_context_length")
                             if isinstance(item.get(key), int) and not isinstance(item.get(key), bool) and item[key] > 0), None)
            if capacity:
                self.model_context_windows[str(item["id"])] = capacity
        return sorted(
            str(item["id"])
            for item in items
            if isinstance(item, dict) and item.get("id")
        )

    # ------------------------------------------------------------------ 请求体
    def _build_payload(self, request: GenerateRequest, *, stream: bool) -> dict:
        payload: dict = {
            "model": request.model or self.model,
            "messages": [],
            "stream": stream,
        }
        options, _ = reasoning_options(self.provider_kind, payload["model"], self.reasoning_effort)
        payload.update(options)
        if request.system:
            payload["messages"].append({"role": "system", "content": request.system})
        thinking = payload.get("thinking", {}).get("type") == "enabled" or payload.get("reasoning_effort") in ("low", "medium", "high", "max")
        reasoning_required = self.provider_kind in ("deepseek", "kimi", "zhipu") or payload["model"].lower().split("/")[-1].startswith(("deepseek", "kimi", "glm"))
        for message in request.messages:
            wire = message.to_wire()
            # With provider/model/effort unchanged, replay depends on this message
            # alone. A new user message, summary instruction, or tool result must
            # not rewrite an earlier assistant and invalidate the shared prefix.
            if thinking and reasoning_required and message.role == "assistant" and message.tool_calls and message.reasoning:
                wire["reasoning_content"] = message.reasoning
            payload["messages"].append(wire)
        if request.tools:
            # 工具 schema 走协议字段,不塞进 system 提示 —— 这是 KV 缓存友好的做法。
            payload["tools"] = [schema.to_wire() for schema in request.tools]
        return payload

    def _request(self, payload: dict) -> urllib.request.Request:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream"
                if payload.get("stream")
                else "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

    # ------------------------------------------------------------------ 非流式
    async def generate(self, request: GenerateRequest) -> GenerateResult:
        payload = self._build_payload(request, stream=False)
        data = await asyncio.to_thread(self._post, payload)
        return parse_completion(data)

    def _post(self, payload: dict) -> dict:
        try:
            with urllib.request.urlopen(self._request(payload), timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise _wrap_transport_exception(exc, self.endpoint, self.timeout) from exc

    # -------------------------------------------------------------------- 流式
    async def stream(self, request: GenerateRequest) -> AsyncIterator[StreamEvent]:
        """SSE 流式生成。

        线程与事件循环的桥:``urllib`` 是同步的,所以让子线程逐行读 SSE,
        用 ``call_soon_threadsafe`` 把每个 chunk 投进 asyncio 队列,这里再变成
        async 生成器。消费方提前退出(取消)时会置 ``stop`` 标志,worker 下一行就收手
        并关闭连接 —— 否则那个 socket 会一直挂到模型把话说完。
        """
        payload = self._build_payload(request, stream=True)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        finished = object()
        state = {"stop": False}

        def worker() -> None:
            try:
                for chunk in self._iter_sse_chunks(payload, state):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except BaseException as exc:  # noqa: BLE001 —— 原样带回消费侧再抛
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, finished)

        worker_future = loop.run_in_executor(None, worker)
        accumulator = StreamAccumulator()
        try:
            while True:
                item = await queue.get()
                if item is finished:
                    break
                if isinstance(item, BaseException):
                    raise item
                for event in accumulator.feed(item):
                    yield event
        finally:
            state["stop"] = True
            try:
                await worker_future
            except Exception:  # noqa: BLE001 —— worker 的异常已在上面带回
                pass

        yield StreamEvent("done", result=accumulator.finish())

    def _iter_sse_chunks(self, payload: dict, state: dict) -> Iterator[dict]:
        """同步 SSE 行读取器:每读到一个 ``data:`` 就 yield 解析后的 dict。

        **解析不了的 ``data:`` 行会被静默跳过**(心跳、半个 chunk、非 JSON 心跳都靠这条)。
        代价要清楚:如果上游/测试脚本给出了非法 JSON,表现是"空响应"而不是报错 ——
        排查空响应时先怀疑这一条。
        """
        response = None
        try:
            response = urllib.request.urlopen(
                self._request(payload), timeout=self.timeout
            )
            for raw_line in response:
                if state.get("stop"):
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue  # 心跳 / 注释
                if not line.startswith("data:"):
                    continue  # event: / id: 等字段迷你版不关心
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue  # 半个 chunk 或非 JSON 心跳:跳过而不是炸掉整条流
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise _wrap_transport_exception(exc, self.endpoint, self.timeout) from exc
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:  # noqa: BLE001
                    pass



def plugin(
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    timeout: float = 120.0,
) -> Plugin:
    """装载 provider 并设为当前模型来源。"""

    def apply(ctx: Context) -> None:
        adapter = OpenAICompatAdapter(api_key, base_url, model, timeout)
        ctx.effect(ctx.llm.register_adapter(adapter.name, adapter))
        ctx.llm.use(adapter.name)

    return Plugin(
        name="llm-openai-compat",
        apply=apply,
        inject=("llm",),
        description="OpenAI 兼容模型适配器(DashScope / DeepSeek / OpenAI 通用)",
    )
