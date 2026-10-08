"""One HTTP UI with independently running conversation contexts."""
from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import replace
from pathlib import Path
import queue
import threading

from .webui_errors import UiConflictError
from .user_messages import user_message_title


class WebUi:
    # Public compatibility attributes refer to the selected conversation. Running
    # coroutines are bound to ConversationUi objects, never this changing facade.
    _LOCAL = frozenset({"_current", "_workers", "_owned_contexts", "_template_config", "runtime_factory",
                        "_cleanup_futures", "_server", "_thread", "_clients", "_action_lock", "loop", "host", "port", "port_note", "_closed", "_archive_stop", "_archive_thread"})

    def __init__(self, ctx, config=None, host="127.0.0.1", port=8770, approval_timeout=300):
        from .webui import ConversationUi
        self._current = ConversationUi(ctx, config, host, port, approval_timeout)
        self._workers = {}
        self._owned_contexts = []
        self._cleanup_futures = []
        self._template_config = config
        self.runtime_factory = None  # Embedders may provide (source_context, cwd) -> fresh Context.
        self.host, self.port, self.port_note = host, port, ""
        self.loop, self._server, self._thread = None, None, None
        self._clients = []
        self._action_lock = threading.RLock()
        self._closed = False
        self._archive_stop = threading.Event()
        self._archive_thread = None

    def __getattr__(self, name):
        current = self.__dict__.get("_current")
        if current is None:
            raise AttributeError(name)
        return getattr(current, name)

    def __setattr__(self, name, value):
        if name in self._LOCAL or hasattr(type(self), name) or "_current" not in self.__dict__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._current, name, value)

    def __delattr__(self, name):
        if name in self.__dict__:
            object.__delattr__(self, name)
        else:
            delattr(self._current, name)

    def attach(self, session=None):
        self._current.attach(session)
        self._adopt(self._current)

    def _adopt(self, worker):
        with self._action_lock:
            for key, existing in list(self._workers.items()):
                if existing is worker:
                    self._workers.pop(key)
            worker.publish = lambda message: self._relay(worker, message)
            worker.bind_loop(self.loop)
            self._workers[worker.session.id] = worker

    def bind_loop(self, loop):
        self.loop = loop
        self._current.bind_loop(loop)
        for worker in self._workers.values():
            worker.bind_loop(loop)

    def start(self):
        from .webui import ConversationUi
        self._closed = False
        address = ConversationUi.start(self)
        self.cleanup_expired_storage()
        if self._archive_thread is None or not self._archive_thread.is_alive():
            self._archive_stop.clear()
            def reap():
                while not self._archive_stop.wait(3600):
                    self.cleanup_expired_storage()
            self._archive_thread = threading.Thread(target=reap, name="storage-cleanup", daemon=True)
            self._archive_thread.start()
        return address

    def cleanup_expired_storage(self):
        """Hourly maintenance also runs while the browser is idle."""
        self.history.cleanup()
        with self._action_lock:
            workers = list(self._workers.values()) or [self._current]
        seen = set()
        for worker in workers:
            store = worker.ctx.get("spillStore", None)
            root = getattr(store, "root", id(store))
            if callable(getattr(store, "cleanup_if_due", None)) and root not in seen:
                store.cleanup_if_due()
                seen.add(root)

    def subscribe(self):
        from .webui import ConversationUi
        return ConversationUi.subscribe(self)

    def unsubscribe(self, target):
        self._clients = [c for c in self._clients if c.queue is not target]

    def publish(self, message):
        from .webui import ConversationUi
        ConversationUi.publish(self, message)

    def _relay(self, worker, message):
        if self._closed:
            return
        kind = message.get("kind")
        if kind == "history":
            self.publish({"kind": "history", "items": self.history_items()})
            return
        # Approval/question cards belong to their conversation. Background
        # workers advertise waiting in history; selecting one restores its cards
        # from snapshot(). Never insert its actionable card into another transcript.
        if worker is self._current:
            self.publish({**message, "session": worker.session.id})
        if kind in {"busy", "user", "approval", "approval-closed", "question", "question-closed"}:
            self.publish({"kind": "history", "items": self.history_items()})

    def history_items(self):
        current = self._current.session
        rows = {row["id"]: row for row in self.history.list(current.source_path if current else None)}
        for worker in list(self._workers.values()):
            session = worker.session
            if session.events_of("session/parent"):
                continue
            users = session.events_of("user/message")
            if not users:
                continue
            key = self.history.key(session.source_path) if session.source_path else "live:" + session.id
            if key not in rows:
                rows[key] = {"id": key, "title": user_message_title(users[0].data),
                             "turns": len(users), "updated_at": users[-1].ts,
                             "cwd": str(worker.ctx.agentLoop.cwd), "workspace_key": str(worker.ctx.agentLoop.cwd)}
            questions = worker.ctx.get("userQuestions", None)
            rows[key].update(session=session.id, active=session is current, running=worker._busy,
                             waiting=bool(worker._approval_messages or (questions and questions.pending)))
        return sorted(rows.values(), key=lambda row: row["updated_at"], reverse=True)

    def status(self):
        return self._current.status()

    def snapshot(self):
        result = self._current.snapshot()
        result["history"] = self.history_items()
        return result

    def target(self, session_id=None):
        if session_id is None:
            return self._current
        if not isinstance(session_id, str):
            raise ValueError("session 必须是对话 ID 字符串")
        worker = self._workers.get(session_id)
        if worker is None:
            raise ValueError("对话已关闭，请刷新后重试")
        return worker

    def _on_loop(self, action):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self.loop is None or not self.loop.is_running() or self.loop is loop:
            return action()
        async def apply():
            return action()
        return asyncio.run_coroutine_threadsafe(apply(), self.loop).result(timeout=30)

    def _fresh_context(self, cwd):
        if self.runtime_factory:
            return self.runtime_factory(self._current.ctx, cwd)
        from .app import HarnessConfig, build_plugins
        from .kernel import Context, Plugin, mount
        template = self._template_config
        if not isinstance(template, HarnessConfig):
            raise RuntimeError("嵌入式 UI 需要配置 runtime_factory 才能创建独立对话运行环境")
        source = self.ctx.llm.active
        clone = getattr(source, "clone_for_conversation", None)
        try:
            adapter = clone() if clone else copy.deepcopy(source)
        except Exception as exc:
            raise RuntimeError("当前模型插件无法复制独立状态，请实现 clone_for_conversation 或配置 runtime_factory") from exc
        config = replace(template, task_cwd=cwd, model=getattr(source, "model", template.model),
                         session_root=self.ctx.sessions.root)
        ctx = Context(label="web-conversation")
        plugins = [p for p in build_plugins(config) if p.name not in {"webui", "llm-openai-compat", "llm-mock"}]
        def provider(target):
            target.effect(target.llm.register_adapter(adapter.name, adapter))
            target.llm.use(adapter.name)
        plugins.append(Plugin("conversation-provider", provider, inject=("llm",)))
        try:
            mount(ctx, plugins)
        except BaseException:
            ctx.dispose()
            raise
        return ctx

    def _create_worker(self, cwd, session=None):
        from .webui import ConversationUi
        ctx = self._fresh_context(cwd)
        worker = None
        try:
            config = copy.copy(self.config)
            if config is not None:
                config.task_cwd = cwd
            worker = ConversationUi(ctx, config, self.host, self.port, self.approval_timeout)
            worker.history, worker.providers = self.history, self.providers
            worker.autosave = self.autosave
            ctx.agentLoop.cwd = cwd
            worker.attach(session)
            with self._action_lock:
                self._adopt(worker)
                self._owned_contexts.append(ctx)
            return worker
        except BaseException:
            # A mounted context is ours even before registration succeeds.
            with self._action_lock:
                for key, registered in list(self._workers.items()):
                    if registered is worker:
                        self._workers.pop(key)
                if ctx in self._owned_contexts:
                    self._owned_contexts.remove(ctx)
            try:
                if worker is not None:
                    worker.stop()
            finally:
                ctx.dispose()
            raise

    def _dispose_owned(self, ctx):
        with self._action_lock:
            if ctx not in self._owned_contexts:
                return
            ctx.dispose()
            self._owned_contexts.remove(ctx)

    def _discard_worker(self, worker):
        """Roll back a prepared, unselected worker (no turn has started)."""
        with self._action_lock:
            self._workers.pop(worker.session.id, None)
            try:
                worker.stop()
            finally:
                self._dispose_owned(worker.ctx)

    def _select(self, worker):
        self._current = worker
        snapshot = self.snapshot()
        self.publish(snapshot)
        return snapshot

    def new_session(self, cwd=None):
        def apply():
            with self._action_lock:
                if self._current._choosing_directory:
                    raise UiConflictError("请先关闭目录选择窗口")
                target = Path(cwd).expanduser().resolve() if cwd else Path(self.ctx.agentLoop.cwd)
                if not target.is_dir():
                    raise ValueError(f"目录不存在: {target}")
                self._current._save_session()
                return self._select(self._create_worker(target))
        return self._on_loop(apply)

    def subagent_log(self, child_id, parent_id=None):
        """Read a completed child's saved log without adopting or resuming it."""
        def apply():
            parent = self.target(parent_id).session
            ends = [e for e in parent.events_of("subagent/end")
                    if e.data.get("child_session") == child_id]
            if not ends:
                raise ValueError("这条子会话不属于当前对话")
            saved = ends[-1].data.get("session_path")
            if not saved:
                raise ValueError("子会话日志未保存，无法查看")
            try:
                path = self.history.resolve(self.history.key(Path(saved)))
                content = path.read_text(encoding="utf-8-sig")
            except (OSError, ValueError) as exc:
                raise ValueError("子会话日志已不存在或无法读取") from exc
            # Verify lineage from the file too; never expose a replaced unrelated log.
            linked = False
            for line in content.splitlines():
                try:
                    event = json.loads(line)
                    if (event.get("type") == "session/parent"
                            and event.get("data", {}).get("parent_session") == parent.id):
                        linked = True
                except (ValueError, AttributeError):
                    continue
            if not linked:
                raise ValueError("子会话日志的父会话标记不匹配")
            return {"title": ends[-1].data.get("title") or "子代理", "content": content}
        return self._on_loop(apply)

    def open_history(self, key):
        def apply():
            with self._action_lock:
                if self._current._choosing_directory:
                    raise UiConflictError("请先关闭目录选择窗口")
                for worker in self._workers.values():
                    session = worker.session
                    known = self.history.key(session.source_path) if session.source_path else "live:" + session.id
                    if key == known:
                        workspaces = session.events_of("session/workspace")
                        cwd = Path(workspaces[-1].data["cwd"]) if workspaces else Path(worker.ctx.agentLoop.cwd)
                        if not cwd.is_dir():
                            raise ValueError(f"目录不存在: {cwd}")
                        return self._select(worker)
                path = self.history.resolve(key)
                # Loading a new worker never repairs/reopens a live conversation's log.
                restored = self.ctx.sessions.open(path)
                workspaces = restored.events_of("session/workspace")
                cwd = Path(workspaces[-1].data["cwd"]).resolve() if workspaces else Path(self.ctx.agentLoop.cwd)
                if not cwd.is_dir():
                    raise ValueError(f"目录不存在: {cwd}")
                self._current._save_session()
                return self._select(self._create_worker(cwd, restored))
        return self._on_loop(apply)

    def delete_history(self, key):
        def apply():
            with self._action_lock:
                path = self.history.resolve(key)
                worker = next((w for w in self._workers.values() if w.session.source_path and w.session.source_path.resolve() == path.resolve()), None)
                if worker and (worker._busy or worker._choosing_directory):
                    raise UiConflictError("该对话正在运行，请先停止它再删除")
                if worker:
                    worker._save_session()
                replacement = self._create_worker(Path(self.ctx.agentLoop.cwd)) if worker is self._current else None
                # Prepare first so a creation failure leaves the old conversation
                # usable; roll back preparation if archival fails instead.
                try:
                    token = self.history.delete(key)
                except BaseException:
                    if replacement is not None:
                        self._discard_worker(replacement)
                    raise
                if worker:
                    self._workers.pop(worker.session.id, None)
                    worker.stop()
                    self._dispose_owned(worker.ctx)
                if replacement:
                    self._current = replacement
                self.publish(self.snapshot())
                return {"ok": True, "undo_token": token}
        return self._on_loop(apply)

    def restore_history(self, token):
        def apply():
            with self._action_lock:
                key = self.history.restore(token)
                self.publish({"kind": "history", "items": self.history_items()})
                return {"ok": True, "id": key}
        return self._on_loop(apply)

    def archived_history(self):
        return {"items": self.history.archives(), "retention_days": self.history.retention_seconds / 86400}

    def purge_history(self, token):
        self.history.purge(token)
        return {"ok": True}

    async def shutdown(self):
        workers = list(self._workers.values())
        for worker in workers:
            worker.interrupt()
        # Only this UI's turns are ours to cancel, not every task on the event loop.
        tasks = [w._turn_task for w in workers if w._turn_task and w._turn_task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        futures = [asyncio.wrap_future(w._turn_future) for w in workers if w._turn_future is not None]
        if futures:
            await asyncio.gather(*futures, return_exceptions=True)
        # stop() may have scheduled disposal before shutdown() was called.
        # Those jobs own the final save and must finish before the loop closes.
        cleanup = list(self._cleanup_futures)
        if cleanup:
            await asyncio.gather(*(asyncio.wrap_future(f) for f in cleanup))
            self._cleanup_futures[:] = [f for f in self._cleanup_futures if f not in cleanup]
        if not self._closed:
            for worker in workers:
                worker._save_session()

    def stop(self):
        self._archive_stop.set()
        if self._closed:
            return
        self._closed = True
        for worker in list(self._workers.values()):
            worker.interrupt()
            worker.stop()
            if worker.ctx in self._owned_contexts:
                task = worker._turn_task
                future = worker._turn_future
                pending = ((task is not None and not task.done()) or
                           (future is not None and not future.done()))
                if pending and self.loop and not self.loop.is_closed():
                    async def dispose_when_idle(target=worker):
                        try:
                            if target._turn_future:
                                await asyncio.gather(asyncio.wrap_future(target._turn_future), return_exceptions=True)
                            pending_task = target._turn_task
                            if pending_task and pending_task is not asyncio.current_task():
                                await asyncio.gather(pending_task, return_exceptions=True)
                            target._save_session()
                        finally:
                            self._dispose_owned(target.ctx)
                    self._cleanup_futures.append(
                        asyncio.run_coroutine_threadsafe(dispose_when_idle(), self.loop))
                else:
                    self._dispose_owned(worker.ctx)
        # Pending contexts stay owned until disposal completes. Embedders must
        # await shutdown() before closing a loop with active turns.
        for client in list(self._clients):
            try:
                client.queue.put_nowait({"kind": "bye"})
            except queue.Full:
                pass
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
