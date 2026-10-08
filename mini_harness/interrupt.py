"""``ctx.interrupt`` —— 协作式取消令牌。

对应 dsh 里 agent loop 的取消路径(dsh 把取消视为"提交承诺"的一部分:取消期间
system 与 user 消息都不提交)。迷你版保留最核心的三条语义:

* **协作式** —— 谁都能 ``request()``,但"真正停下"由循环与工具在检查点决定,
  不靠抛异常打断;
* **可复位** —— 一个 turn 被取消后 ``reset()``,REPL 里下一个 turn 照常;
* **可从信号处理器调用** —— CLI 的 SIGINT 处理器直接调 ``request()``。
  这里刻意**不抛异常**:抛出去会打断 asyncio 的清理路径,会话日志就残了
  (而日志是唯一真相,残了下一个 turn 的请求就是非法的)。

为什么用两个 Event:``threading.Event`` 提供线程安全的标志位(信号处理器里也能用),
``asyncio.Event`` 给 ``await token.wait()`` 用 —— 工具靠它和取消赛跑。
令牌可先后用于不同事件循环,重新绑定时会重建 asyncio.Event;
不支持同时由多个运行中的事件循环共享。request() 可从其他线程调用,
bind_loop()/wait()/reset() 由使用令牌的事件循环线程管理。
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from .kernel import Context, Plugin

__all__ = ["InterruptService", "InterruptRecord", "plugin"]

DEFAULT_REASON = "用户中断"


@dataclass
class InterruptRecord:
    """一次取消请求。留给日志与 UI 读。"""

    reason: str = DEFAULT_REASON
    source: str = "user"
    extra: dict[str, Any] = field(default_factory=dict)


class InterruptService:
    """取消令牌。对应 ``ctx.interrupt``。"""

    def __init__(self) -> None:
        self._record: InterruptRecord | None = None
        self._flag = threading.Event()
        self._async_event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._listeners: list[Callable[[InterruptRecord], None]] = []

    # ------------------------------------------------------------------ 状态
    @property
    def cancelled(self) -> bool:
        return self._flag.is_set()

    @property
    def reason(self) -> str | None:
        return self._record.reason if self._record else None

    @property
    def record(self) -> InterruptRecord | None:
        return self._record

    def on_request(self, listener: Callable[[InterruptRecord], None]) -> Callable[[], None]:
        """预留的取消订阅接口,当前生产路径未订阅。

        回调在 request() 的调用线程/信号上下文中同步执行,应快速返回且不抛异常。
        需要写日志等操作的消费者应将工作调度回所属事件循环。
        """
        self._listeners.append(listener)

        def dispose() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return dispose

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """由驱动器在进入 turn 时绑定当前事件循环,使信号处理器能安全唤醒等待方。"""
        if self._loop is not loop:
            if self._loop is not None and self._loop.is_running():
                raise RuntimeError("取消令牌不能同时绑定多个运行中的事件循环")
            # Event keeps its first waiting loop internally; changing only _loop
            # would update wake routing but leave wait() bound to the old loop.
            self._async_event = None
        self._loop = loop

    # ------------------------------------------------------------------ 动作
    def request(self, reason: str = DEFAULT_REASON, source: str = "user") -> bool:
        """发起取消。返回 True 表示这是**第一次**请求(重复请求返回 False)。

        可以从信号处理器里调用:只用线程安全原语,不抛异常。
        """
        if self._flag.is_set():
            return False
        self._record = InterruptRecord(reason=reason, source=source)
        self._flag.set()
        if self._async_event is not None:
            self._wake()
        for listener in list(self._listeners):
            listener(self._record)
        return True

    def reset(self) -> None:
        """复位,准备下一个 turn。"""
        self._record = None
        self._flag.clear()
        if self._async_event is not None:
            self._async_event.clear()

    async def wait(self) -> str:
        """等到被取消为止,返回取消原因。供工具与取消赛跑。"""
        self.bind_loop(asyncio.get_running_loop())
        if self._async_event is None:
            self._async_event = asyncio.Event()
        if self._flag.is_set():
            self._async_event.set()
        await self._async_event.wait()
        return self.reason or DEFAULT_REASON

    # ------------------------------------------------------------------ 内部
    def _wake(self) -> None:
        event = self._async_event
        if event is None:
            return  # Rebinding may have cleared it after request() checked.
        loop = self._loop
        if loop is not None and loop.is_running():
            # 信号处理器与事件循环同线程,但 call_soon_threadsafe 走 self-pipe,
            # 在任何线程/信号上下文中都安全。
            def wake_if_requested():
                # A queued wake from the previous turn must not cancel a new
                # turn after reset() has already cleared the token.
                if self._flag.is_set() and self._async_event is event and self._loop is loop:
                    event.set()
            try:
                loop.call_soon_threadsafe(wake_if_requested)
            except RuntimeError:
                if not loop.is_closed():
                    raise
                # Closing a loop can race with a request from another thread.
                # The flag survives and the next wait() observes it.
        # No running loop: keep the flag; wait() applies it on the owner thread.


def plugin() -> Plugin:
    def apply(ctx: Context) -> None:
        ctx.provide("interrupt", InterruptService())

    return Plugin(
        name="interrupt",
        apply=apply,
        description="协作式取消令牌",
    )
