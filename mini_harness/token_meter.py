"""``ctx.tokenMeter`` —— 上下文窗口与用量。

对应 dsh 的 `dsh-token-meter`。三个数各有各的来路,**别混**:

* **窗口大小** —— 显式配置优先，其次使用网关 ``/models`` 返回的原始容量；
  网关未提供容量时使用近似表和默认值，不将报告容量取整或截断。
  界面上会把这个数标成"未知/近似",百分比只给量级感,不是合同;
* **估算用量** —— 组装请求时按字符估(复用压缩的 ``estimate_tokens``，2 字符/token)。
  可能高估也可能低估，并非分词器的精确结果；
* **实测用量** —— provider 返回的 ``usage.prompt_tokens``,这才是权威值。
  把两个数并排显示还有个好处:估算误差**一眼可见**,不用猜。

快照描述当前请求：新请求开始时清空旧响应的实测/输出/缓存字段。
终端消费这份快照；网页的“上次输入/缓存”和摘要统计从持久化 usage 读取，
以便请求在途或重新打开会话时仍可回查。两者是同一 usage 的不同时间视图，
缓存字段不参与估算校准，也不能从计量器的 requests 推导总调用次数或账单。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .kernel import MODE_EMIT, Context, Plugin

__all__ = [
    "UsageSnapshot",
    "TokenMeter",
    "window_for",
    "DEFAULT_WINDOW",
    "plugin",
]

#: 近似窗口表(**顺序敏感**:靠前的先匹配,所以更具体的写在前面)。
#:
#: 这是只按模型名称估算的回退表，不代表本次网关上报的容量。
#: 压缩与显示共用窗口信息；未知容量使用近似值，可显式配置覆盖。
# DeepSeek entries follow the advisory catalogue in llm-deepseek/defaults.ts.
# Other models use the requested 256K fallback, explicitly marked as estimated.
_MODEL_WINDOWS: tuple[tuple[str, int], ...] = (
    ("deepseek-v4", 1_000_000),
    ("deepseek-flash", 1_000_000),
    ("qwen-turbo", 1_000_000),
)
DEFAULT_WINDOW = 256_000

#: 新样本占 0.5：每次向经过限幅的样本移动一半距离，较快适应内容类型变化。
#: 这会减弱而非消除单次样本的影响；不用于识别或剔除异常样本。
SMOOTHING = 0.5


def cache_usage(usage: dict[str, Any] | None) -> tuple[int | None, float | None]:
    """Chat Completions cache-read tokens; absence is unknown, never a zero hit.

    Supports OpenAI prompt_tokens_details and DeepSeek prompt_cache_hit_tokens.
    Input total includes cached tokens; no price/savings inference is made here.
    """
    if not isinstance(usage, dict):
        return None, None
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    if cached is None:
        cached = usage.get("prompt_cache_hit_tokens")
    prompt = usage.get("prompt_tokens")
    valid = lambda value: isinstance(value, int) and not isinstance(value, bool) and value >= 0
    if not valid(cached) or (valid(prompt) and cached > prompt):
        return None, None
    return cached, (100 * cached / prompt if valid(prompt) and prompt > 0 else None)


def window_for(model: str) -> tuple[int, bool]:
    """按名称查近似表或兜底值，返回 ``(估算窗口大小, True)``。

    本函数不知道显式配置和 provider 上报，第二项始终为 True。
    完整优先级由 TokenMeter.window 处理，来源由 window_source 区分。
    保留二元组返回形式，以兼容现有调用方。
    """
    name = (model or "").lower()
    for keyword, size in _MODEL_WINDOWS:
        if keyword in name:
            return size, True
    return DEFAULT_WINDOW, True


@dataclass
class UsageSnapshot:
    """当前请求的用量视图；不是历史累计或费用统计。

    measured/output_tokens/cached_tokens/cache_hit_percent 未报告时为 None。
    estimated 驱动 calibrated/used；终端使用缓存字段，网页历史值另读日志。
    """

    model: str = ""
    window: int = DEFAULT_WINDOW
    estimated: int = 0
    measured: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    cache_hit_percent: float | None = None
    requests: int = 0  # 本实例收到非空 usage 的次数（包括恢复时的重新加载）。
    window_is_guess: bool = True
    #: 估算 × 校准系数 = 更接近真实的估计值
    factor: float = 1.0

    @property
    def calibrated(self) -> int:
        return int(self.estimated * self.factor)

    @property
    def used(self) -> int:
        """优先用实测值;还没有响应时用**校准后**的估算。"""
        if self.measured is not None:
            return self.measured
        return self.calibrated or self.estimated

    @property
    def percent(self) -> float:
        if self.window <= 0:
            return 0.0
        return min(100.0, self.used / self.window * 100)


class TokenMeter:
    """记住"最近一次请求有多大",供 UI 显示。对应 ``ctx.tokenMeter``。"""

    def __init__(self, model: str = "", window: int | None = None) -> None:
        self._model = model
        self._window = int(window) if window else 0
        self._provider_windows: dict[str, int] = {}
        self._estimated = 0
        self._measured: int | None = None
        self._output: int | None = None
        self._cached: int | None = None
        self._cache_percent: float | None = None
        self._requests = 0
        self._factor = 1.0

    # ------------------------------------------------------------------ 状态
    @property
    def model(self) -> str:
        return self._model

    @property
    def window(self) -> int:
        if self._window > 0:
            return self._window
        reported = self._provider_windows.get(self._model)
        if reported:
            return reported
        return window_for(self._model)[0]

    @property
    def window_source(self) -> str:
        if self._window > 0:
            return "configured"
        return "provider" if self._model in self._provider_windows else "estimated"

    @property
    def window_is_guess(self) -> bool:
        return self._window <= 0 and self._model not in self._provider_windows

    def clear_capacities(self) -> None:
        self._provider_windows.clear()

    def note_capacities(self, capacities: dict[str, int]) -> None:
        self._provider_windows.update({
            name: size for name, size in capacities.items()
            if isinstance(size, int) and not isinstance(size, bool) and size > 0
        })

    def set_model(self, name: str) -> None:
        """换模型:实测值与校准系数一起作废。

        实测值是上一个模型报的;校准系数也一样 —— 不同模型的分词器不一样,
        拿旧比例去估新模型只会更偏。
        """
        self._model = name
        self._measured = None
        self._output = None
        self._cached, self._cache_percent = None, None
        self._factor = 1.0

    @property
    def factor(self) -> float:
        """估算校准系数(实测/估算 的滑动平均)。1.0 = 还没校过。"""
        return self._factor

    # ------------------------------------------------------------------ 记录
    def note_request(self, tokens: int) -> None:
        """发请求之前:记下估算值,并把**上一轮的实测值作废**。

        为什么不留着:实测值属于上一个请求,把它搭在新请求的估算旁边会自相矛盾
        (估算按新内容算,实测还是老的份额)。清掉之后语义就干净了 ——
        请求在途期间显示校准后的估算,收到响应再换成实测,两个数各自对得上自己那一轮。
        """
        self._estimated = max(0, int(tokens))
        self._measured = None
        self._output = None
        self._cached, self._cache_percent = None, None

    def note_response(self, usage: dict[str, Any] | None) -> None:
        """收到响应之后:记下权威值(provider 说的才算),并用它校准估算。"""
        if not usage:
            return
        self._cached, self._cache_percent = cache_usage(usage)
        self._requests += 1
        prompt = usage.get("prompt_tokens")
        if isinstance(prompt, int) and not isinstance(prompt, bool) and prompt >= 0:
            self._measured = prompt
            self._update_factor(prompt)
        completion = usage.get("completion_tokens")
        if isinstance(completion, int) and not isinstance(completion, bool) and completion >= 0:
            self._output = completion

    def _update_factor(self, measured: int) -> None:
        """用实测纠正"字符 → token"的比例。

        为什么要校:``estimate_tokens`` 用的是 2 字符/token 这个折中值
        (中文接近 1–2,ASCII 接近 4),英文代码场景会高估近一倍。
        先把本次 实测/估算 样本限制到 [0.2, 5.0]，再做指数平滑。
        factor 从 1.0 起步，是旧值与有界样本的凸组合，因此也保持在该范围；
        并没有再次显式夹取 factor。SMOOTHING=0.5 会保留样本一半的影响，
        用较快响应换取对单次噪声较弱的抑制。
        """
        if self._estimated <= 0:
            return
        sample = min(5.0, max(0.2, measured / self._estimated))
        self._factor = round((1 - SMOOTHING) * self._factor + SMOOTHING * sample, 4)

    def snapshot(self) -> UsageSnapshot:
        return UsageSnapshot(
            model=self._model,
            window=self.window,
            estimated=self._estimated,
            measured=self._measured,
            output_tokens=self._output,
            cached_tokens=self._cached,
            cache_hit_percent=self._cache_percent,
            requests=self._requests,
            window_is_guess=self.window_is_guess,
            factor=self._factor,
        )


def plugin(model: str = "", window: int | None = None) -> Plugin:
    """装载用量表,并让它跟着模型切换走(``llm/model-changed``)。"""

    def apply(ctx: Context) -> None:
        meter = TokenMeter(model, window)
        ctx.provide("tokenMeter", meter)
        ctx.on("llm/model-capacities", meter.note_capacities, mode=MODE_EMIT)

        def on_model_changed(name: str, previous: str) -> None:  # noqa: ARG001
            meter.set_model(name)

        ctx.effect(ctx.on("llm/model-changed", on_model_changed, mode=MODE_EMIT))

    return Plugin(
        name="token-meter",
        apply=apply,
        description="上下文窗口与用量统计",
    )
