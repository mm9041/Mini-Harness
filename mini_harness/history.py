"""Disk-backed conversation catalogue; API clients select opaque IDs, never paths."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import shutil
import uuid
import time
import math
import threading
from functools import wraps
from pathlib import Path
from .user_messages import user_message_title


def _locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


class ConversationHistory:
    retention_seconds = 7 * 24 * 60 * 60

    def __init__(self, root: Path):
        self._lock = threading.RLock()
        self.root = Path(root)
        self.index = self.root / ".history-index.json"
        self._external: set[str] = set()
        self._cache: dict = {}
        if self.index.is_file():
            try:
                self._external = {p for p in json.loads(self.index.read_text(encoding="utf-8")) if isinstance(p, str)}
            except (OSError, ValueError, TypeError):
                pass

    @staticmethod
    def key(path: Path) -> str:
        return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()

    @_locked
    def remember(self, path: Path) -> None:
        path = path.resolve()
        if path.parent == self.root.resolve() or str(path) in self._external:
            return
        updated = self._external | {str(path)}
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=self.root, prefix=".history-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(sorted(updated), stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.index)
            self._external = updated
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @_locked
    def paths(self) -> dict[str, Path]:
        paths = {p.resolve() for p in self.root.glob("*.jsonl")}
        paths.update(Path(p) for p in self._external)
        return {self.key(p): p for p in paths if p.is_file()}

    @_locked
    def resolve(self, key: str) -> Path:
        path = self.paths().get(key)
        if path is None:
            raise ValueError("找不到这条历史对话，请刷新列表")
        return path

    @_locked
    def delete(self, key: str) -> str:
        """Move one registered conversation to recoverable local storage."""
        source = self.resolve(key).resolve()
        summary = next((item for item in self.list() if item["id"] == key), {})
        token = uuid.uuid4().hex
        folder = self._archive_folder(token)
        folder.mkdir(parents=True)
        (folder / "metadata.json").write_text(
            json.dumps({"original": str(source), "archived_at": time.time(),
                        "title": summary.get("title", "未命名对话"), "cwd": summary.get("cwd", "")}, ensure_ascii=False), encoding="utf-8"
        )
        shutil.move(str(source), str(folder / "conversation.jsonl"))
        self._cache.pop(key, None)
        return token

    def _archive_folder(self, token: str) -> Path:
        if len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
            raise ValueError("无效的恢复标识")
        trash = self.root.resolve() / ".trash"
        folder = trash / token
        if trash.resolve() != trash or folder.resolve() != folder:
            raise ValueError("归档路径不能是链接")
        return folder

    def _archive_data(self, folder: Path) -> dict:
        metadata = folder / "metadata.json"
        if metadata.is_symlink():
            raise ValueError("无效的归档元数据")
        data = json.loads(metadata.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("无效的归档元数据")
        # Older archives have no timestamp; metadata was created when deleted.
        data.setdefault("archived_at", metadata.stat().st_mtime)
        data["archived_at"] = float(data["archived_at"])
        if not math.isfinite(data["archived_at"]):
            raise ValueError("无效的归档时间")
        data["expires_at"] = data["archived_at"] + self.retention_seconds
        return data

    @_locked
    def archives(self) -> list[dict]:
        self.cleanup()
        items = []
        for candidate in (self.root.resolve() / ".trash").glob("*"):
            try:
                folder = self._archive_folder(candidate.name)
                data = self._archive_data(folder)
                archived = folder / "conversation.jsonl"
                if not archived.is_file() or archived.is_symlink():
                    continue
                title = data.get("title")
                if not title:
                    with archived.open(encoding="utf-8-sig") as stream:
                        for line in stream:
                            try:
                                event = json.loads(line)
                                if event.get("type") == "user/message":
                                    title = user_message_title(event["data"])
                                    break
                            except (ValueError, KeyError, TypeError, AttributeError):
                                continue
                items.append({"token": candidate.name, "title": title or "未命名对话",
                              "cwd": data.get("cwd", ""), "archived_at": data["archived_at"],
                              "expires_at": data["expires_at"]})
            except (OSError, ValueError, TypeError):
                continue
        return sorted(items, key=lambda item: float(item["archived_at"]), reverse=True)

    @_locked
    def purge(self, token: str) -> None:
        folder = self._archive_folder(token)
        if not folder.is_dir():
            raise ValueError("归档已经删除或不存在")
        shutil.rmtree(folder)

    @_locked
    def cleanup(self, now: float | None = None) -> list[str]:
        now = time.time() if now is None else now
        removed = []
        for candidate in (self.root.resolve() / ".trash").glob("*"):
            try:
                folder = self._archive_folder(candidate.name)
                if now >= self._archive_data(folder)["expires_at"]:
                    self.purge(candidate.name)
                    removed.append(candidate.name)
            except (OSError, ValueError, TypeError):
                continue
        return removed

    @_locked
    def restore(self, token: str) -> str:
        folder = self._archive_folder(token)
        try:
            data = self._archive_data(folder)
            if time.time() >= data["expires_at"]:
                self.purge(token)
                days = self.retention_seconds / 86400
                raise ValueError(f"这条归档已超过保留期（{days:g} 天）")
            original = Path(data["original"])
            archived = folder / "conversation.jsonl"
            if not archived.is_file() or archived.is_symlink():
                raise ValueError("这条对话已经恢复或不存在")
            if original.exists():
                raise ValueError("原路径已有文件，无法覆盖恢复")
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(archived), str(original))
            self.remember(original)
            self.purge(token)
            return self.key(original)
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError("找不到可恢复的对话") from exc

    @_locked
    def list(self, current: Path | None = None, *, include_children: bool = False) -> list[dict]:
        items = []
        for key, path in self.paths().items():
            try:
                stat = path.stat()
                fingerprint = (stat.st_mtime_ns, stat.st_size)
                cached = self._cache.get(key)
                if cached and cached[0] == fingerprint:
                    item = dict(cached[1])
                else:
                    title, count, cwd, updated_at = "", 0, "", 0.0
                    is_child = False
                    with path.open(encoding="utf-8-sig") as stream:
                        for line in stream:
                            try:
                                event = json.loads(line)
                                data = event.get("data") or {}
                                if event.get("type") == "session/parent":
                                    is_child = True
                                if event.get("type") == "user/message":
                                    count += 1
                                    stamp = event.get("ts")
                                    if isinstance(stamp, (int, float)):
                                        updated_at = max(updated_at, stamp)
                                    if not title:
                                        title = user_message_title(data)
                                if event.get("type") == "session/workspace":
                                    cwd = data.get("cwd", "")
                            except (ValueError, TypeError, AttributeError):
                                continue
                    item = {"id": key, "title": title or "未命名对话", "turns": count, "is_child": is_child,
                            "updated_at": updated_at or stat.st_mtime, "cwd": cwd,
                            "workspace_key": os.path.normcase(os.path.normpath(cwd)) if cwd else ""}
                    self._cache[key] = (fingerprint, dict(item))
                # Keep child logs discoverable for inspection, outside the chat sidebar.
                if not item["turns"] or (item["is_child"] and not include_children):
                    continue
                item["active"] = current is not None and path.resolve() == current.resolve()
                items.append(item)
            except OSError:
                continue
        return sorted(items, key=lambda item: item["updated_at"], reverse=True)
