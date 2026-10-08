"""命令行入口。

对应 dsh 的 ``dsh --profile headless "task"``(单次执行)与交互模式。三种跑法:

* ``python -m mini_harness "任务"`` —— 跑一个 turn 就退出;
* ``python -m mini_harness`` (TTY 下)或 ``--repl`` —— 进入多轮交互,复用同一个会话;
* ``python -m mini_harness --mock "任务"`` —— 离线,不调用真实模型。

配置来源:命令行参数 > 已存在的环境变量 > ``.env`` > 内置默认值。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
from collections import Counter
from pathlib import Path

from . import __version__, builtin_tools
from .adapters.openai_compat import resolve_chat_endpoint
from .app import AUTO_ENV_FILE, DEFAULT_PERSONA, HarnessConfig, build_context
from .approval import ApprovalDecision, ApprovalRequest
from .console import TracePrinter, status_for
from .console_input import read_line
from .kernel import MODE_EMIT, MountError, ServiceNotFound
from .llm import LLMError
from .repl import ReplSession

__all__ = ["main", "build_parser"]

MOCK_COMMAND = "echo mini-harness-ok"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini_harness",
        description=(
            "一个可运行的迷你 agent harness:插件树 + 工具 + 会话日志 + turn/step 循环"
            "(支持流式输出与多轮交互)。默认对接 DashScope 的 OpenAI 兼容模式。"
        ),
    )
    parser.add_argument("task", nargs="?", help="交给 agent 的任务描述;不给则进入交互模式")
    parser.add_argument("--repl", action="store_true", help="强制进入多轮交互模式")
    parser.add_argument(
        "--web", action="store_true", help="启动网页版 UI(标准库 HTTP + SSE,零依赖)"
    )
    parser.add_argument("--host", help="网页版监听地址(默认 127.0.0.1)")
    parser.add_argument("--port", type=int, help="网页版端口(默认 8770;0 = 随机端口)")
    parser.add_argument("--open", action="store_true", help="启动后用默认浏览器打开")
    parser.add_argument(
        "--resume", help="续跑一个既有会话(JSONL 路径):读回日志后接着往下走"
    )
    parser.add_argument(
        "--continue",
        dest="continue_session",
        action="store_true",
        help="续跑最近一次会话(会话目录里最新的那个 JSONL)",
    )
    parser.add_argument("--cwd", help="工作目录(默认当前目录),shell 与文件工具在此执行")
    parser.add_argument(
        "--model", help="模型名(默认 $DASHSCOPE_MODEL,再退回 qwen-plus)"
    )
    parser.add_argument(
        "--api-key", help="API key(默认读 $DASHSCOPE_API_KEY / $DEEPSEEK_API_KEY / $OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--base-url",
        help="OpenAI 兼容端点(默认 $DASHSCOPE_BASE_URL 或 DashScope 官方地址)",
    )
    parser.add_argument("--timeout", type=float, help="单次模型请求超时秒数(默认 120)")
    parser.add_argument("--max-steps", type=int, help="单轮模型决策步数上限(默认 64，每步可调用多个工具)")
    parser.add_argument(
        "--no-stream", action="store_true", help="关闭流式输出(等整段返回再打印)"
    )
    parser.add_argument(
        "--no-early-tools",
        action="store_true",
        help="关闭「边流边执行」:工具参数补全后不再提前派发",
    )
    parser.add_argument("--permission", choices=("read-only", "workspace-write", "danger-full-access"), help="权限预设（默认 workspace-write）")
    parser.add_argument(
        "--approval",
        choices=("ask", "allow", "deny"),
        help="已弃用，请使用 --permission",
    )
    parser.add_argument(
        "--allow-outside", action="store_true", help="已弃用，请使用 --permission"
    )
    parser.add_argument(
        "--no-spill",
        action="store_true",
        help="关闭工具结果外溢(超预算退回硬截断,内容拿不回来)",
    )
    parser.add_argument(
        "--spill-root", help="外溢目录(默认 ~/.mini-harness/spill)"
    )
    parser.add_argument(
        "--no-compaction", action="store_true", help="关闭上下文压缩"
    )
    parser.add_argument(
        "--history-budget",
        type=int,
        help="额外历史 token 上限(默认 0，按模型窗口自动压缩)",
    )
    parser.add_argument(
        "--no-subagents",
        action="store_true",
        help="不给模型 task 工具(关掉子代理委派)",
    )
    parser.add_argument(
        "--subagent-depth",
        type=int,
        help="允许的子代理嵌套深度(默认 1 = 只派一层)",
    )
    parser.add_argument(
        "--retry",
        type=int,
        help="模型调用失败时的总尝试次数(默认 3;0 = 不重试)",
    )
    parser.add_argument(
        "--retry-always",
        action="store_true",
        help="无限重试,直到成功或取消(对应 dsh 的 always 模式)",
    )
    parser.add_argument(
        "--retry-delay", type=float, help="指数退避的基数秒数(默认 0.5)"
    )
    parser.add_argument(
        "--context-window",
        type=int,
        help="上下文窗口大小(用于状态行的百分比;默认按模型名查近似表,网关不返回这个数)",
    )
    parser.add_argument("--shell", help="指定 shell:bash / pwsh / powershell / cmd 或完整路径")
    parser.add_argument("--shell-timeout", type=float, help="单条命令超时秒数(默认 60)")
    parser.add_argument("--session-root", help="会话 JSONL 落盘目录")
    parser.add_argument("--save-session", help="把会话事件写成 JSONL 到指定路径")
    parser.add_argument("--dump-events", action="store_true", help="在输出末尾打印会话 JSONL")
    parser.add_argument("--persona", help="覆盖系统人格提示")
    parser.add_argument(
        "--env-file", help="指定 .env 路径(默认先找当前目录,再找项目根目录)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help=f"离线模式:用脚本化适配器代替真实模型(执行 {MOCK_COMMAND!r}),不需要 API key",
    )
    parser.add_argument("--list-tools", action="store_true", help="打印模型可见的工具清单后退出")
    parser.add_argument("--quiet", action="store_true", help="不打印 turn/step 事件轨迹")
    parser.add_argument("--version", action="version", version=f"mini-harness {__version__}")
    return parser


# ---------------------------------------------------------------------- 配置


def _build_config(args: argparse.Namespace, offline: bool | None = None) -> HarnessConfig:
    """命令行参数交给 ``HarnessConfig.from_env`` 覆盖环境变量与 .env。"""
    if args.approval is not None or args.allow_outside:
        raise ValueError("独立审批/路径开关已由权限预设取代，请使用 --permission read-only|workspace-write|danger-full-access")
    env_file: object = AUTO_ENV_FILE
    if args.env_file:
        env_file = Path(args.env_file).expanduser().resolve()

    return HarnessConfig.from_env(
        env_file=env_file,
        task_cwd=Path(args.cwd).expanduser().resolve() if args.cwd else Path.cwd(),
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        max_steps=args.max_steps,
        timeout=args.timeout,
        shell_timeout=args.shell_timeout,
        shell=args.shell,
        session_root=(
            Path(args.session_root).expanduser().resolve()
            if args.session_root
            else None
        ),
        persona=args.persona or DEFAULT_PERSONA,
        streaming=False if args.no_stream else None,
        early_tools=False if args.no_early_tools else None,
        permission_preset=args.permission,
        approval=args.approval,
        allow_outside=True if args.allow_outside else None,
        spill=False if args.no_spill else None,
        spill_root=args.spill_root,
        compaction=False if args.no_compaction else None,
        max_history_tokens=args.history_budget,
        context_window=args.context_window,
        subagents=False if args.no_subagents else None,
        subagent_max_depth=args.subagent_depth,
        retry_attempts=args.retry,
        retry_always=True if args.retry_always else None,
        retry_base_delay=args.retry_delay,
        offline=offline,
    )


# ---------------------------------------------------------------------- 输出


def _write(text: str) -> None:
    """统一出口:不换行、立即刷 —— 流式与交互提示都要能立刻看到。"""
    sys.stdout.write(text)
    sys.stdout.flush()


def _print_banner(config: HarnessConfig, session_id: str) -> None:
    print(f"[harness] 会话 {session_id} | 工作目录 {config.task_cwd}")
    _print_shell_line(config)
    if config.offline:
        print("[harness] 模型 离线脚本化适配器(不调用真实 API)")
        return

    try:
        endpoint = resolve_chat_endpoint(config.base_url)
    except LLMError:
        endpoint = config.base_url
    print(f"[harness] 模型 {config.model} → {endpoint}")
    if config.api_key_source:
        print(f"[harness] key 来自 {config.api_key_source}")
    if config.env_file:
        print(f"[harness] 配置来自 {config.env_file}")


def _print_mode_line(config: HarnessConfig) -> None:
    stream = "流式输出" if config.streaming else "整段返回"
    early = " · 边流边执行" if (config.streaming and config.early_tools) else ""
    approval = {
        "ask": "操作前询问",
        "allow": "自动批准",
        "deny": "拒绝受控操作",
    }.get(config.approval, config.approval)
    if config.permission_preset:
        approval = config.permission_preset
    spill = f"外溢 ≤{config.spill_max_inline_chars} 字符" if config.spill else "外溢 关"
    compaction = (
        (f"压缩 {config.max_history_tokens} tok" if config.max_history_tokens else f"压缩 窗口 {config.compaction_threshold_ratio:.0%}") if config.compaction else "压缩 关"
    )
    subagents = f"子代理 ≤{config.subagent_max_depth} 层" if config.subagents else "子代理 关"
    if config.retry_attempts <= 0:
        retry = "重试 关"
    elif config.retry_always:
        retry = "重试 无限"
    else:
        retry = f"重试 ≤{config.retry_attempts} 次"
    print(
        f"[harness] 模式 {stream}{early} · 权限 {approval}"
        f" · {spill} · {compaction} · {retry} · {subagents}"
    )
    if config.spill:
        print(f"[harness] 外溢目录 {_short_path(config)}")


def _short_path(config: HarnessConfig) -> str:
    """外溢目录:用 ~ 缩写主目录前缀,一行放得下。"""
    from . import spill

    root = Path(config.spill_root).expanduser() if config.spill_root else spill.default_root()
    try:
        return f"~/{root.relative_to(Path.home())}".replace("\\", "/")
    except ValueError:
        return str(root)


def _print_shell_line(config: HarnessConfig) -> None:
    """把选中的 shell 打出来 —— 命令跑在哪个 shell 里,是排查环境问题第一条线索。"""
    import shutil
    executable = shutil.which("pwsh") or (shutil.which("powershell") if os.name == "nt" else None)
    print(f"[harness] pwsh 后端 PowerShell ({executable or '未安装'})")


# ------------------------------------------------------------ 审批与中断接线


def _install_approver(ctx) -> None:
    """把 ``ctx.approval`` 的"怎么问人"接到终端上。

    审批策略(哪些要问)在插件里,问法在 UI 里 —— 这里就是 UI 那一半。
    """
    service = ctx.get("approval", None)
    if service is None:
        return

    def approver(request: ApprovalRequest) -> ApprovalDecision:
        detail = json.dumps(request.call.arguments or {}, ensure_ascii=False)
        if len(detail) > 500:
            detail = detail[:500] + " …"
        _write(
            f"\n⚠ 需要审批:{request.reason}\n"
            f"  工具:{request.call.name}\n"
            f"  参数:{detail}\n"
            f"  允许执行?[y/N] "
        )
        # 审批提示期间让 Ctrl+C 恢复默认行为(抛 KeyboardInterrupt),
        # 否则我们那个"只设令牌不抛异常"的处理器会让用户按了没反应。
        previous = signal.signal(signal.SIGINT, signal.default_int_handler)
        try:
            answer = input()
        except (EOFError, KeyboardInterrupt):
            _write("\n")
            ctx.interrupt.request("审批阶段被中断")
            return ApprovalDecision(False, "用户在审批阶段中断")
        finally:
            signal.signal(signal.SIGINT, previous)

        approved = answer.strip().lower() in ("y", "yes", "是", "ok")
        return ApprovalDecision(
            approved, "" if approved else "用户在审批时选择了拒绝"
        )

    service.set_approver(approver)


def _install_interrupt_handler(ctx) -> None:
    """Ctrl+C → 协作式取消,而不是把进程连日志一起打断。

    第一次 Ctrl+C 只设令牌(循环会在检查点收尾,日志保持完整合法);
    第二次 Ctrl+C 才硬退。
    """
    token = ctx.get("interrupt", None)
    if token is None or not hasattr(signal, "SIGINT"):
        return

    def handler(signum, frame):  # noqa: ARG001
        if token.request("用户按了 Ctrl+C"):
            _write("\n[中断] 已请求取消当前 turn,等它收尾…\n")
        else:
            _write("\n[中断] 再次 Ctrl+C,强制退出。\n")
            raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, handler)
    except ValueError:
        # 非主线程(例如被测试调用)装不了信号处理器,忽略即可。
        pass


def _install_question_responder(ctx) -> None:
    service = ctx.get("userQuestions", None)
    if service is None or not _stdin_is_tty():
        return

    async def respond(message):
        _write("\n需要你的回答：" + message["question"] + "\n")
        for index, option in enumerate(message["options"], 1):
            _write(f"  {index}. {option}\n")
        try:
            while True:
                answer = (await read_line("输入回答或选项序号（/skip 跳过）：")).strip()
                if answer == "/skip":
                    return None
                if answer.isdigit() and 1 <= int(answer) <= len(message["options"]):
                    return message["options"][int(answer) - 1]
                if answer and len(answer) <= 10000:
                    return answer
        except (EOFError, KeyboardInterrupt):
            ctx.interrupt.request("用户提问阶段被中断")
            return None
    service.responder = respond


# ---------------------------------------------------------------------- 主流程


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.approval is not None or args.allow_outside:
        parser.error("独立审批/路径开关已由 --permission 权限预设取代")

    if args.list_tools:
        return _print_tools(args)

    want_repl = args.repl or (not args.task and _stdin_is_tty())
    if not args.task and not want_repl and not args.web:
        parser.error("请给一个任务描述,或用 --repl 进入交互模式,或用 --list-tools 看工具清单")

    config = _build_config(args, offline=True if args.mock else None)

    try:
        ctx = build_context(config)
    except (LLMError, MountError, ServiceNotFound, RuntimeError, ValueError, OSError) as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        return 2

    _install_interrupt_handler(ctx)
    _install_approver(ctx)
    _install_question_responder(ctx)

    try:
        session, notes = _make_session(ctx, args)
    except (OSError, ValueError) as exc:
        ctx.dispose()
        print(str(exc), file=sys.stderr)
        return 2
    for note in notes:
        print(f"[会话] {note}")

    try:
        if args.web:
            return _run_web(ctx, config, args, session)
        if want_repl:
            return _run_repl(ctx, config, args, session)
        return _run_once(ctx, config, args, session)
    finally:
        ctx.dispose()


def _make_session(ctx, args: argparse.Namespace):
    """建会话,或续跑一个既有会话。返回 ``(session, 说明行)``。

    续跑只做两件事:**读回日志** + **补齐未闭合的尾巴**。没有别的魔法 ——
    模型历史本来就从日志投影出来,所以"接着上次说"不需要额外的状态。
    """
    wanted: Path | None = None

    if args.resume:
        wanted = Path(args.resume).expanduser()
        if not wanted.is_file():
            raise FileNotFoundError(f"找不到要续跑的会话文件:{wanted}")
    elif args.continue_session:
        wanted = ctx.sessions.latest()
        if wanted is None:
            return ctx.sessions.create(), ["会话目录里没有可续的会话,已新建一个"]

    if wanted is None:
        return ctx.sessions.create(), []

    session = ctx.sessions.open(wanted)
    notes = [f"续跑 {wanted}(既有 {len(session.events)} 条事件)"]
    notes.extend(f"修复:{note}" for note in session.repairs)
    return session, notes


def _stdin_is_tty() -> bool:
    try:
        return bool(sys.stdin and sys.stdin.isatty())
    except (ValueError, OSError):
        return False


def _run_once(
    ctx, config: HarnessConfig, args: argparse.Namespace, session
) -> int:
    printer = TracePrinter(
        write=_write, show_trace=not args.quiet, show_stream=config.streaming
    )
    dispose_observer = printer.observe(session)
    dispose_stream = printer.attach(ctx)

    if not args.quiet:
        _print_banner(config, session.id)
        _print_mode_line(config)
    print(status_for(ctx, config))

    agent = ctx.agents.create(session)
    try:
        result = asyncio.run(agent.run(args.task))
    except LLMError as exc:
        print(f"\n模型调用失败: {exc}", file=sys.stderr)
        _persist(session, args)
        return 3
    except KeyboardInterrupt:
        print("\n已强制退出。", file=sys.stderr)
        _persist(session, args)
        return 130
    finally:
        dispose_stream()
        dispose_observer()

    streamed = printer.show_stream and printer.streamed_last_step
    print("\n=== 最终回答 ===")
    if streamed:
        print("(已在上方流式输出,此处不重复;加 --no-stream 可改为整段打印)")
    else:
        print(result.text or "(模型没有返回文本)")

    counts = Counter(event.type for event in session.events)
    print(
        f"\n[会话] {session.id} | step={result.steps} | 停止原因={result.stopped} "
        f"| 用时 {result.duration_ms / 1000:.1f}s | 事件数={len(session.events)}"
    )
    print("        " + ", ".join(f"{name}×{count}" for name, count in sorted(counts.items())))
    print(status_for(ctx, config))  # 收尾再画一次:这时的用量是实测值

    saved = _persist(session, args)
    if config.offline:
        print("\n提示:这是离线演示。去掉 --mock 就会走真实模型。")
    return 0 if saved else 2


def _run_web(ctx, config: HarnessConfig, args: argparse.Namespace, session=None) -> int:
    """网页版:主线程跑事件循环,HTTP 请求线程往里调度协程。

    为什么不是"每个请求起一个 asyncio.run":那样每次都是一棵新的插件树、一个新的会话,
    "多轮对话"就没了。这里刻意让**一个循环、一个会话**贯穿始终 ——
    和其他几种跑法用同一个模型:会话日志是唯一真相,前端只是它的一个视图。
    """
    from . import webui as webui_module

    host = args.host or config.webui_host
    port = args.port if args.port is not None else config.webui_port

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ui = webui_module.serve(ctx, config, host, port, loop, session, args.save_session)
    url = f"http://{ui.host}:{ui.port}/"
    print(f"[harness] 网页版已启动:{url}")
    print(f"[harness] 模型 {config.model} · 工作目录 {config.task_cwd}")
    print("[harness] Ctrl+C 停止服务(页面上的「停止」按钮只中断当前这一轮)")
    if args.task:
        print("[harness] 提示:网页模式下任务参数被忽略,请在页面里输入")
    if args.open:
        import webbrowser

        webbrowser.open(url)

    # 网页模式下 Ctrl+C 的语义是"停服务",不是"取消这一轮"(取消请点页面上的按钮)
    def stop(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print("[harness] 停止服务…")
    finally:
        loop.run_until_complete(ui.shutdown())
        ui.stop()
        loop.close()
    return 0


def _run_repl(
    ctx, config: HarnessConfig, args: argparse.Namespace, session
) -> int:
    printer = TracePrinter(
        write=_write, show_trace=not args.quiet, show_stream=config.streaming
    )
    repl = ReplSession(ctx, config, printer=printer, session=session)
    _print_banner(config, repl.session_id)
    _print_mode_line(config)
    try:
        code = repl.start()
    finally:
        saved = _persist(repl.session, args)
    return code if code or saved else 2


def _persist(session, args: argparse.Namespace) -> bool:
    """落盘:显式给了路径就写那里;**续跑过的会话默认写回原文件**。"""
    try:
        _persist_once(session, args)
        return True
    except Exception as exc:
        # Called during error handling/finally too: preserve the original failure.
        print(f"会话保存失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False


def _persist_once(session, args: argparse.Namespace) -> None:
    if args.save_session:
        target: Path | None = Path(args.save_session).expanduser()
    elif session.source_path is not None:
        target = session.source_path
    else:
        target = None

    if target is not None:
        print(f"[已保存] {session.save(target)}")
    elif args.dump_events:
        print("\n=== 会话事件(JSONL)===")
        print(session.to_jsonl())


def _print_tools(args: argparse.Namespace) -> int:
    config = _build_config(args, offline=True)  # 看工具清单不需要真实模型
    try:
        ctx = build_context(config)
    except (LLMError, MountError, ServiceNotFound, RuntimeError, ValueError, OSError) as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        return 2
    schemas = ctx.tools.schemas()  # type: ignore[attr-defined]
    print(f"模型可见工具 {len(schemas)} 个 (命令工具: pwsh / PowerShell):")
    for schema in schemas:
        print(f"\n- {schema.name}: {schema.description}")
        print("  parameters: " + json.dumps(schema.parameters, ensure_ascii=False))
    return 0
