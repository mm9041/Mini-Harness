"""交互式多轮 REPL —— 复用同一个 Session。

这是"日志是唯一真相"第一次真正显出价值的地方:每一轮都往**同一个** append-only
日志里追加,而模型历史每一步都从日志重新投影 —— 所以第二轮天然知道第一轮说过什么、
工具干过什么,**不需要任何"对话缓存"这种东西**。

两个实现细节值得留意:

* 所有 turn 用**同一个事件循环**(``loop.run_until_complete``),而不是每个 turn 开一个
  ``asyncio.run``:避免反复创建、关闭事件循环,让会话运行时保持一致。
  取消令牌本身也支持在先后使用不同事件循环时重新绑定。
* 提示符处按 Ctrl+C **不清空会话、也不退出**,只是提示怎么退出;turn 进行中按 Ctrl+C
  由信号处理器转成协作式取消(见 ``interrupt.py``),本轮收尾后回到提示符。
"""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from typing import Callable

from .approval import ACCESS_LABELS
from .console import TracePrinter, status_for
from .llm import LLMError

__all__ = ["ReplSession", "HELP_TEXT"]

HELP_TEXT = """可用命令(其他任何输入都当作任务交给 agent):
  /help              显示本帮助
  /model [名称]      看或换模型(不带名称 = 列出网关可用模型)
  /permission [预设] 查看或切换 read-only / workspace-write / danger-full-access
  /access [档位]     看或换权限档位:ask=操作前询问 · full=自动批准 · deny=拒绝受控操作
  /tools             列出模型可见的工具
  /events            打印当前会话的事件统计
  /compact           把较早的历史压成摘要(省 token,立即生效并落日志)
  /sessions          列出最近的会话文件(可续跑)
  /resume [路径|序号] 切换到某个既有会话接着聊(不带参数 = 最近一个)
  /save [路径]       把会话事件写成 JSONL(默认写回它的来源文件)
  /exit  /quit       退出(等价于 Ctrl+D)"""

#: 权限档位的别名 —— 允许直接说人话。
_ACCESS_ALIASES = {
    "ask": "ask",
    "approve": "ask",
    "批准": "ask",
    "请求批准": "ask",
    "full": "allow",
    "allow": "allow",
    "完全访问": "allow",
    "deny": "deny",
    "readonly": "deny",
    "只读": "deny",
}


class ReplSession:
    """一次交互会话。会话对象、agent 句柄、事件循环都跨轮复用。"""

    def __init__(
        self,
        ctx,
        config=None,
        printer: TracePrinter | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        input_fn: Callable[[str], str] = input,
        write: Callable[[str], None] | None = None,
        prompt: str = "mini-harness> ",
        session=None,
    ) -> None:
        self.ctx = ctx
        self.config = config  # 仅供参考(CLI 用它打 banner),REPL 自己不读
        self.printer = printer or TracePrinter()
        self.loop = loop or asyncio.new_event_loop()
        self.input_fn = input_fn
        self.write = write or (lambda text: print(text, end="", flush=True))
        self.prompt = prompt

        self.session = session or ctx.sessions.create()
        self.agent = ctx.agents.create(self.session)
        self.turns = 0
        #: 最近一次 /sessions 列出的文件,供 /resume 序号引用。
        self._recent: list[Path] = []

        self._dispose_observer = self.printer.observe(self.session)
        self._dispose_stream = self.printer.attach(ctx)

    # ------------------------------------------------------------------ 生命周期
    @property
    def session_id(self) -> str:
        return self.session.id

    def _workspace(self) -> Path:
        """相对路径的落脚点:优先配置里的工作目录,退回进程 cwd。"""
        task_cwd = getattr(self.config, "task_cwd", None) if self.config else None
        return Path(task_cwd) if task_cwd else Path.cwd()

    def status_line(self) -> str:
        """实时状态行:模型 / 工作目录 / 上下文窗口与用量 / 权限。

        读的是活状态(``ctx.tokenMeter`` / ``ctx.approval``),所以 ``/model`` 与
        ``/access`` 切完之后立刻反映出来。
        """
        return status_for(self.ctx, self.config, self.session)

    def _current_model(self) -> str:
        """正在用的模型名:优先问**适配器**(它才是真相),其次用量表。

        先读用量表/配置会显示一个其实没在用的名字 —— 离线适配器就忽略配置里的模型名。
        """
        service = self.ctx.get("llm", None)
        adapter = getattr(service, "active", None) if service is not None else None
        live = getattr(adapter, "model", "")
        if live:
            return live
        meter = self.ctx.get("tokenMeter", None)
        return getattr(meter, "model", "") or "(未知)"

    def switch_session(self, path: Path) -> None:
        """切换到另一个既有会话接着聊。

        切走之前,如果当前会话是从文件读回来的,就写回原文件 —— 免得"换了个会话,
        上一个会话的这几轮悄悄没了"。新会话的 agent 句柄与观察者都要重新挂。
        """
        if self.session.events and self.session.source_path is not None:
            saved = self.ctx.sessions.save(self.session)
            self.write(f"(已把当前会话写回 {saved})\n")

        restored = self.ctx.sessions.open(path)
        self._dispose_observer()
        self.session = restored
        self.agent = self.ctx.agents.create(self.session)
        self._dispose_observer = self.printer.observe(self.session)
        self.turns = 0

        self.write(f"已切换到会话 {self.session.id}({len(self.session.events)} 条事件)\n")
        for note in self.session.repairs:
            self.write(f"  修复:{note}\n")

    def close(self) -> None:
        self._dispose_stream()
        self._dispose_observer()

    def start(self) -> int:
        resumed = (
            f"(续跑,已有 {len(self.session.events)} 条事件)"
            if self.session.events
            else ""
        )
        self.write(
            f"进入交互模式,会话 {self.session.id}{resumed}。"
            "输入 /help 看命令,Ctrl+D 或 /exit 退出。\n"
        )
        try:
            while True:
                # 状态行每次进提示符前重画一遍 —— 这样每轮之后用量都是新的。
                self.write(self.status_line() + "\n")
                try:
                    line = self.input_fn(self.prompt)
                except EOFError:
                    self.write("\n")
                    break
                except KeyboardInterrupt:
                    self.write("\n(会话仍在,用 Ctrl+D 或 /exit 退出)\n")
                    continue

                line = (line or "").strip()
                if not line:
                    continue

                handled = self.handle_command(line)
                if handled is False:
                    break
                if handled is True:
                    continue
                self.run_turn(line)
        finally:
            self.close()
        return 0

    # ------------------------------------------------------------ 斜杠命令
    def handle_command(self, line: str) -> bool | None:
        """True = 已作为命令处理(继续);False = 该退出;None = 不是命令,当任务跑。"""
        try:
            return self._handle_command(line)
        except (OSError, ValueError) as exc:
            self.write(f"命令失败:{type(exc).__name__}: {exc}\n当前会话仍保留在内存中，可继续操作或另存为其他路径。\n")
            return True

    def _handle_command(self, line: str) -> bool | None:
        head, _, rest = line.partition(" ")
        command = head.lower()

        if command in ("/exit", "/quit", ":q"):
            self.write("再见。\n")
            return False

        if command == "/help":
            self.write(HELP_TEXT + "\n")
            return True

        if command == "/model":
            self._handle_model(rest.strip())
            return True

        if command == "/access":
            self._handle_access(rest.strip())
            return True

        if command == "/tools":
            schemas = self.ctx.tools.schemas()
            self.write(f"模型可见工具 {len(schemas)} 个:\n")
            for schema in schemas:
                first_line = (schema.description or "").splitlines()[0]
                self.write(f"  - {schema.name}: {first_line}\n")
            return True

        if command == "/events":
            counts = Counter(event.type for event in self.session.events)
            self.write(f"会话 {self.session.id} 共 {len(self.session.events)} 条事件:\n")
            for name, count in sorted(counts.items()):
                self.write(f"  {name}: {count}\n")
            self.write(
                f"  turn {counts.get('turn/start', 0)} 个,"
                f"step {counts.get('step/start', 0)} 个,"
                f"工具调用 {counts.get('tool/call', 0)} 次\n"
            )
            return True

        if command == "/permission":
            permissions = self.ctx.get("permissions", None)
            if permissions is None:
                self.write("权限预设未启用。\n")
                return True
            try:
                target = line.strip().split(maxsplit=1)[1] if len(line.strip().split(maxsplit=1)) > 1 else ""
                if target:
                    permissions.set(target, self.session)
                self.write(f"当前权限预设：{permissions.current(self.session)}\n")
            except ValueError as exc:
                self.write(str(exc) + "\n")
            return True

        if command == "/compact":
            service = self.ctx.get("compaction", None)
            if service is None:
                self.write("压缩没有启用(启动时加了 --no-compaction?)。\n")
                return True
            self.write(service.manual_retention_hint() + "\n")
            try:
                record = self.loop.run_until_complete(
                    service.condense_now(self.session)
                )
            except Exception as exc:  # noqa: BLE001 —— 压不动不该把 REPL 弄崩
                self.write(f"压缩失败:{type(exc).__name__}: {exc}\n")
                return True

            if record is None:
                self.write("没有可压缩的余量(消息太少,或没什么可折叠的)。\n")
                return True
            self.write(
                f"已把 {record.covered_messages} 条消息压成摘要:"
                f"≈{record.tokens_before} tok → ≈{record.tokens_after} tok"
                f"(省 ≈{record.saved_tokens} tok)。\n"
            )
            if record.saved_tokens <= 0:
                self.write(
                    "注意:这次摘要没省下 token —— 历史本来就短,摘要反而更长。"
                    "自动压缩只在真有压力、且要压的内容够多时才会做。\n"
                )
            self.write("摘要已写进会话日志(compaction 事件),续跑之后仍然生效。\n")
            return True

        if command == "/sessions":
            self._recent = self.ctx.sessions.list_sessions(limit=10)
            if not self._recent:
                self.write(
                    f"会话目录里还没有文件({self.ctx.sessions.root})。"
                    "用 /save 或 --save-session 落盘后就能在这里看到。\n"
                )
                return True
            self.write(f"最近的会话({self.ctx.sessions.root}):\n")
            for position, path in enumerate(self._recent, start=1):
                lines = sum(
                    1
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
                mark = " ←当前" if path == self.session.source_path else ""
                self.write(f"  [{position}] {path.name}  {lines} 条事件{mark}\n")
            self.write("用 /resume <序号|路径> 切换过去。\n")
            return True

        if command == "/resume":
            target = self._resolve_session_reference(rest.strip())
            if target is None:
                return True
            self.switch_session(target)
            return True

        if command == "/save":
            raw = rest.strip()
            if raw:
                candidate = Path(raw).expanduser()
                # 相对路径落在工作目录里,而不是"启动进程时的 cwd" —— 后者常和
                # 用户以为的工作区不是一回事(比如用了 --cwd)。
                target = (
                    candidate
                    if candidate.is_absolute()
                    else self._workspace() / candidate
                )
            else:
                target = None
            saved = self.ctx.sessions.save(self.session, target)
            self.write(f"已保存 {saved}\n")
            return True

        if line.startswith("/"):
            self.write(f"未知命令 {head};输入 /help 看可用命令。\n")
            return True

        return None

    def _handle_model(self, name: str) -> None:
        """``/model``:不带参数列出来,带参数就换掉。"""
        if not name:
            self.write(f"当前模型:{self._current_model()}\n")
            try:
                names = self.loop.run_until_complete(self.ctx.llm.list_models())
            except Exception as exc:  # noqa: BLE001 —— 网关不支持就算了
                self.write(f"(拿不到模型列表:{type(exc).__name__}: {exc})\n")
                self.write("用法:/model <模型名>\n")
                return
            self.write(f"网关可用 {len(names)} 个模型:\n")
            for index in range(0, min(len(names), 60), 3):
                row = "  ".join(f"{item:<28}" for item in names[index : index + 3])
                self.write(f"  {row.rstrip()}\n")
            if len(names) > 60:
                self.write(f"  …(还有 {len(names) - 60} 个未列出)\n")
            self.write("用 /model <名称> 切换。\n")
            return

        # 最低限度的校验:模型名不含空白、不以 / 开头。
        # 少了这一道,把 "/access" 这种**命令**误当模型名也会被默默接受(实测踩到:
        # 一行里写了两个命令,于是模型被改成了 "/access")。
        if name.startswith("/") or any(char.isspace() for char in name):
            self.write(f"{name!r} 看起来不是模型名;先用 /model 看可用列表。\n")
            return

        try:
            previous = self.loop.run_until_complete(self.ctx.llm.use_model(name))
        except Exception as exc:  # noqa: BLE001
            self.write(f"换模型失败:{type(exc).__name__}: {exc}\n")
            return
        self.write(f"模型 {previous} → {name}(下一个 step 生效;提示词里的模型名已同步)\n")

    def _handle_access(self, want: str) -> None:
        """``/access``:看或换权限档位。"""
        service = self.ctx.get("approval", None)
        if service is None:
            self.write("审批服务没装载:启动时用了 --approval?\n")
            return

        if not want:
            current = ACCESS_LABELS.get(service.mode_for(self.session), service.mode_for(self.session))
            self.write(f"当前权限:{current}\n")
            for mode, label in ACCESS_LABELS.items():
                self.write(f"  /access {mode:<6} → {label}\n")
            return

        mode = _ACCESS_ALIASES.get(want.lower())
        if mode is None:
            self.write(f"未知档位 {want!r};可选 {' / '.join(ACCESS_LABELS)}\n")
            return

        permissions = self.ctx.get("permissions", None)
        if permissions is not None:
            preset = {"ask": "workspace-write", "allow": "danger-full-access", "deny": "read-only"}[mode]
            permissions.set(preset, self.session)
            self.write(f"当前权限预设：{preset}\n")
            return
        previous = service.set_mode(mode, self.session)
        if previous == mode:
            self.write(f"已经是「{ACCESS_LABELS[mode]}」了。\n")
            return
        self.write(
            f"权限 {ACCESS_LABELS[previous]} → {ACCESS_LABELS[mode]}(在途审批已撤销，后续调用按新策略执行)\n"
        )
        if mode == "allow":
            self.write("自动批准受控操作；文件路径范围不变，命令执行没有操作系统沙箱。\n")

    def _resolve_session_reference(self, raw: str) -> Path | None:
        """把 ``/resume`` 的参数解析成一个会话文件:序号、路径,或"最近一个"。"""
        if not raw:
            latest = self.ctx.sessions.latest()
            if latest is None:
                self.write("会话目录里没有可续的会话。\n")
                return None
            return latest

        if raw.isdigit():
            if not self._recent:
                self._recent = self.ctx.sessions.list_sessions(limit=10)
            index = int(raw) - 1
            if not 0 <= index < len(self._recent):
                self.write(f"序号 {raw} 超出范围;先执行 /sessions 看看有哪些。\n")
                return None
            return self._recent[index]

        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self._workspace() / candidate
        if not candidate.is_file():
            self.write(f"找不到会话文件:{candidate}\n")
            return None
        return candidate

    # ------------------------------------------------------------------ 跑一轮
    def run_turn(self, line: str):
        """跑一个 turn。返回 RunResult;被中断或失败时返回 None。"""
        self.ctx.interrupt.reset()
        try:
            result = self.loop.run_until_complete(self.agent.run(line))
        except LLMError as exc:
            self.write(f"\n模型调用失败: {exc}\n")
            return None
        except KeyboardInterrupt:
            # 只在"第二次 Ctrl+C"时才会走到这里(第一次已转成协作式取消)
            self.write("\n[已强制退出当前 turn]\n")
            self.ctx.interrupt.reset()
            self._drain_pending()
            return None

        self.turns += 1
        self.write(
            f"\n=== 第 {self.turns} 轮回答(step={result.steps},"
            f"停止原因={result.stopped},用时 {result.duration_ms / 1000:.1f}s)===\n"
        )
        if not (self.printer.show_stream and self.printer.streamed_last_step):
            self.write((result.text or "(模型没有返回文本)") + "\n")
        if result.stopped == "cancelled":
            self.write(
                "(本轮被中断;已经产生的工具结果仍留在会话日志里,"
                "所以下一轮的历史是完整合法的)\n"
            )
        return result

    def _drain_pending(self) -> None:
        """清掉循环里残留的 task,免得下一轮报 "loop is already running"。"""
        pending = [task for task in asyncio.all_tasks(self.loop) if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            self.loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
