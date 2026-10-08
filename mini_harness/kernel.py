"""迷你版 Cordis 内核 —— dsh "everything is a plugin" 的骨架。

dsh 的每一个部件(模型适配器、工具注册表、会话日志、agent loop 本身)都是插件,都挂在
Cordis 的 context 上。本文件只实现读懂 dsh 架构所必需的五件事,刻意省略
`parallel` / `bail` 两种调度模式与 HMR:

  1. plugin 是实现了 ``apply(ctx)`` 的对象,可选声明 ``inject``;
  2. context 是服务的仓库,服务以稳定 key 注册(``ctx.llm`` / ``ctx.tools`` ...);
  3. 依赖用 ``inject`` 声明,装载顺序由依赖推导,而不是手写顺序;
  4. 事件按调度模式分发:``emit`` / ``waterfall`` / ``serial``;
  5. 注册是可撤销的效果,``ctx.effect()`` 返回 disposer,``dispose()`` 逆序回滚。

对应 dsh 源码:``vendor/cordis``(读 ``docs/cordis-primer.md`` 对照本文件最省力)。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "Context",
    "Plugin",
    "MountError",
    "ServiceNotFound",
    "mount",
    "MODE_EMIT",
    "MODE_WATERFALL",
    "MODE_SERIAL",
]

MODE_EMIT = "emit"
MODE_WATERFALL = "waterfall"
MODE_SERIAL = "serial"

_MISSING = object()


class ServiceNotFound(KeyError):
    """请求了当前 context 树上不存在的服务(对应 Cordis 里 inject 未满足)。"""


class MountError(RuntimeError):
    """插件装载失败:依赖永远无法满足,或服务 key 冲突。"""


class Context:
    """服务的仓库 + 事件总线 + 可撤销注册表。

    服务查找沿父链向上,所以子 context(``ctx.scope()``)可以覆盖父级的服务,
    这正是 dsh 里"给某一个 agent 一套不同的能力集"的实现基础。
    当前生产装配使用独立 root Context;scope 与 serial 保留为扩展接口。
    """

    def __init__(self, parent: "Context | None" = None, label: str = "root") -> None:
        self.parent = parent
        self.label = label
        self._services: dict[str, Any] = {}
        # event -> [(mode, listener)],按注册顺序
        self._listeners: dict[str, list[tuple[str, Callable[..., Any]]]] = {}
        # Shared across this context tree; counts allow mode changes after all
        # registrations for an event have been undone.
        self._event_modes: dict[str, tuple[str, int]] = parent._event_modes if parent is not None else {}
        self._disposers: list[Callable[[], None]] = []
        self._children: list["Context"] = []
        self._disposed = False

    # ------------------------------------------------------------------ 服务

    def provide(self, key: str, value: Any) -> Callable[[], None]:
        """注册一个服务并返回它的注销器。key 在本 context 内必须唯一。"""
        if key in self._services:
            raise MountError(f"服务 {key!r} 已由 {self._services[key]!r} 注册,不能重复提供")
        self._services[key] = value

        def dispose() -> None:
            if self._services.get(key) is value:
                self._services.pop(key, None)

        return self.effect(dispose)

    def has(self, key: str) -> bool:
        """沿父链判断服务是否可见。"""
        node: Context | None = self
        while node is not None:
            if key in node._services:
                return True
            node = node.parent
        return False

    def get(self, key: str, default: Any = _MISSING) -> Any:
        node: Context | None = self
        while node is not None:
            if key in node._services:
                return node._services[key]
            node = node.parent
        if default is not _MISSING:
            return default
        raise ServiceNotFound(
            f"服务 {key!r} 未注册;当前可见: {sorted(self.available_keys())}"
        )

    def available_keys(self) -> set[str]:
        keys: set[str] = set()
        node: Context | None = self
        while node is not None:
            keys |= set(node._services)
            node = node.parent
        return keys

    def __getattr__(self, item: str) -> Any:
        """支持 ``ctx.llm`` 这种写法(对应 Cordis 的 ctx.<serviceKey>)。"""
        if item.startswith("_"):
            raise AttributeError(item)
        try:
            return self.get(item)
        except ServiceNotFound as exc:
            raise AttributeError(str(exc)) from exc

    def scope(self, label: str) -> "Context":
        """派生子 context:能看见父级服务,自己的注册与父级隔离。"""
        child = Context(self, label)
        self._children.append(child)
        return child

    # ------------------------------------------------------- 可撤销的注册效果

    def effect(self, disposer: Callable[[], None]) -> Callable[[], None]:
        """登记一个可撤销效果,返回"手动撤销"函数。dispose() 时会逆序回滚。"""
        if self._disposed:
            disposer()
            return disposer
        self._disposers.append(disposer)

        def undo() -> None:
            if disposer in self._disposers:
                self._disposers.remove(disposer)
                disposer()

        return undo

    def on(
        self,
        event: str,
        listener: Callable[..., Any],
        mode: str = MODE_EMIT,
    ) -> Callable[[], None]:
        """订阅事件。mode 是事件的公开契约的一部分,同一事件只能有一种模式。"""
        if mode not in (MODE_EMIT, MODE_WATERFALL, MODE_SERIAL):
            raise ValueError(f"未知的事件调度模式: {mode!r}")
        fixed, count = self._event_modes.get(event, (mode, 0))
        if fixed != mode:
            raise MountError(f"事件 {event!r} 已按 {fixed!r} 注册,不能再以 {mode!r} 注册")
        self._event_modes[event] = (mode, count + 1)
        entry = (mode, listener)
        self._listeners.setdefault(event, []).append(entry)

        def dispose() -> None:
            bucket = self._listeners.get(event)
            if bucket and entry in bucket:
                bucket.remove(entry)
                registered_mode, remaining = self._event_modes[event]
                if remaining == 1:
                    self._event_modes.pop(event)
                else:
                    self._event_modes[event] = (registered_mode, remaining - 1)

        return self.effect(dispose)

    # ------------------------------------------------------------------ 事件

    def _collect(self, event: str, mode: str) -> list[Callable[..., Any]]:
        """收集监听器:子 context 优先,再沿父链向上(越具体越先执行)。"""
        result: list[Callable[..., Any]] = []
        node: Context | None = self
        while node is not None:
            for entry_mode, listener in node._listeners.get(event, ()):
                if entry_mode != mode:
                    raise RuntimeError(
                        f"事件 {event!r} 以 {entry_mode!r} 注册,却以 {mode!r} 分发;"
                        "调度模式是事件的公开契约,注册与分发必须一致"
                    )
                result.append(listener)
            node = node.parent
        return result

    async def emit(self, event: str, *args: Any) -> None:
        """通知型事件:按注册顺序依次 await,忽略返回值。"""
        for listener in self._collect(event, MODE_EMIT):
            await _invoke(listener, *args)

    async def serial(self, event: str, *args: Any) -> Any:
        """串行型事件:按注册顺序 await,返回最后一个监听器的结果。"""
        result: Any = None
        for listener in self._collect(event, MODE_SERIAL):
            result = await _invoke(listener, *args)
        return result

    async def waterfall(self, event: str, *args: Any, default: Any = None,
                        terminal: Callable[..., Any] | None = None) -> Any:
        """环绕中间件:监听器收到 ``(*args, next)``,调 ``next()`` 才委托给下游。

        不调 ``next()`` 直接返回即为短路(单决策事件的设计用法);只做观察/标注的
        监听器必须委托。``default`` 是链条走完后的兜底值 —— 若希望监听器能"就地改写
        再委托",把待改写的可变对象同时作为参数和 default 传进来即可。
        若监听器会替换参数对象，可提供 terminal；链尾以最新参数调用它，
        而不是返回最初的 default。未提供时保持原有语义。
        """
        listeners = self._collect(event, MODE_WATERFALL)

        async def run(index: int, current_args: tuple[Any, ...]) -> Any:
            if index >= len(listeners):
                return await _invoke(terminal, *current_args) if terminal is not None else default
            listener = listeners[index]

            async def nxt(*new_args: Any) -> Any:
                return await run(index + 1, new_args if new_args else current_args)

            return await _invoke(listener, *current_args, nxt)

        return await run(0, args)

    # ---------------------------------------------------------------- 生命周期

    def dispose(self) -> None:
        """逆序回滚所有注册(先子后父),对应 Cordis 的卸载语义。"""
        for child in list(self._children):
            child.dispose()
        self._children.clear()
        while self._disposers:
            self._disposers.pop()()
        self._listeners.clear()
        self._disposed = True


async def _invoke(listener: Callable[..., Any], *args: Any) -> Any:
    """统一调用:同步监听器直接调,协程监听器 await。"""
    result = listener(*args)
    if inspect.isawaitable(result):
        result = await result
    return result


# ---------------------------------------------------------------------- 插件


@dataclass
class Plugin:
    """一个插件 = 一个名字 + 一次 ``apply(ctx)`` + 可选的依赖声明。

    dsh 里插件的依赖通过 ``inject`` 声明,装载顺序因此由依赖关系推导出来,
    而不是靠人工排序 —— 这是"没有特权核心"能成立的前提。
    """

    name: str
    apply: Callable[[Context], Any]
    inject: Sequence[str] = field(default_factory=tuple)
    description: str = ""


def mount(ctx: Context, plugins: Iterable[Plugin]) -> Context:
    """按依赖顺序把插件装载到 ctx 上。

    反复扫描待装载列表,每一轮只装载依赖已就绪的插件;某一轮若毫无进展,
    说明存在无法满足的依赖(拼错 key 或循环依赖),直接报错而不是静默死锁。
    本函数可向已有 context 增量装载,失败不自动撤销已有注册;
    创建全新 context 的调用方应在失败时 dispose(),例如 app.build_context。
    """
    pending = list(plugins)
    mounted: list[str] = []
    while pending:
        progressed = False
        for plugin in list(pending):
            missing = [key for key in plugin.inject if not ctx.has(key)]
            if missing:
                continue
            pending.remove(plugin)
            plugin.apply(ctx)
            mounted.append(plugin.name)
            progressed = True
        if not progressed:
            detail = {p.name: [k for k in p.inject if not ctx.has(k)] for p in pending}
            raise MountError(
                "以下插件的依赖永远无法满足(检查 key 拼写或循环依赖): "
                + ", ".join(f"{name} 缺少 {keys}" for name, keys in detail.items())
            )
    return ctx
