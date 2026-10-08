"""测试用的公共工具。

目前只有一件事:**在 Windows 上稳妥地删掉临时目录**。

原因(实测):`asyncio` 子进程刚退出时,它的工作目录引用不会立刻被内核释放 ——
虽然只是毫秒级窗口,但机器负载高时会拉长到足以让 `shutil.rmtree` 撞上
``PermissionError: [WinError 32] 另一个程序正在使用此文件``。那个报错出现在测试的
``asyncTearDown`` 里,会让**断言全过**的测试被判为 error。

所以清理不能"一次不成就报错",而要么重试,要么放弃。这里是重试。
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

__all__ = ["make_temp_dir", "remove_tree", "build_context_with", "adapter_plugin_for"]

#: 默认插件树里**由 provider 提供的**那几个 —— 测试要用自己的适配器替换它们。
#:
#: 之前这里按 ``name.startswith("llm-")`` 过滤,结果把 ``llm-retry`` 一起过滤掉了
#: (它名字带 llm- 但不是 provider)。教训:过滤要按**明确的集合**,不要按前缀猜 ——
#: 前缀约定迟早会有例外,而例外表现为"功能静默消失"。
PROVIDER_PLUGIN_NAMES = frozenset({"llm-openai-compat", "llm-mock"})


def make_temp_dir(prefix: str = "mini-harness-test-") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def adapter_plugin_for(adapter):
    """把一个**既有**的适配器实例挂成插件。

    比 ``scripted_plugin`` 多一步的意义:测试能拿到那个实例,断言它究竟收到了哪些请求。
    """
    from mini_harness.kernel import Plugin

    def apply(ctx) -> None:
        ctx.effect(ctx.llm.register_adapter(adapter.name, adapter))
        ctx.llm.use(adapter.name)

    return Plugin(name="llm-adapter", apply=apply, inject=("llm",))


async def remove_tree(
    path: Path | str,
    attempts: int = 30,
    delay: float = 0.05,
) -> bool:
    """尽力删除目录树,返回是否删成功。

    退避策略:固定 ``delay`` 起步、最多放大到 5 倍,总等待上限约 2 秒 —— 足够覆盖
    子进程退出后的句柄释放窗口,又不会让测试卡住。真删不掉就返回 ``False``
    (临时目录留给系统清理),而不是让测试报错。
    """
    target = Path(path)
    for attempt in range(attempts):
        try:
            shutil.rmtree(target)
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            if attempt == attempts - 1:
                return False
            await asyncio.sleep(delay * min(attempt + 1, 5))
    return False


def build_context_with(
    adapter_plugin,
    cwd: Path,
    max_steps: int = 8,
    **config_overrides,
):
    """用**默认插件树**,但把模型 provider 换成给定的离线适配器。

    默认树包含 shell / read_file / write_file / approval / interrupt / agent-loop,
    所以这里测到的接线与线上完全一致;只是模型那一层被换掉了。
    """
    from mini_harness.app import HarnessConfig, build_plugins
    from mini_harness.kernel import Context, mount

    overrides = {
        # Existing component tests exercise the legacy approval seam without a native runtime.
        # Preset integration tests explicitly enable the production default.
        "permission_preset": None,
        # 测试绝不能把外溢文件写进用户主目录 —— 默认落到测试自己的临时目录里
        "spill_root": str(Path(cwd) / "spill"),
        # Sessions, provider profiles and job logs must never touch the real workspace.
        "session_root": Path(cwd) / "sessions",
        **config_overrides,
    }
    config = HarnessConfig(task_cwd=cwd, max_steps=max_steps, offline=True, **overrides)
    plugins = [
        plugin
        for plugin in build_plugins(config)
        if plugin.name not in PROVIDER_PLUGIN_NAMES
    ]
    plugins.append(adapter_plugin)

    ctx = Context(label="test")
    mount(ctx, plugins)
    return ctx
