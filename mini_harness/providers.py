"""Local provider profiles and provider-specific reasoning options.

Windows keys use user-bound DPAPI; file protection also depends on directory ACLs.
chmod(0o600) is not an ACL boundary on Windows. Other platforms store plaintext
keys in a mode-0600 file. DPAPI is randomized: compare unsealed keys, not ciphertext.
Unreadable configuration fails explicitly; corrupt content is backed up before reset.
"""
from __future__ import annotations

import base64
import ctypes
import json
import os
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

EFFORTS = ("none", "low", "medium", "high", "max")
PRESETS = [
    {"id": "deepseek", "name": "DeepSeek", "base_url": "https://api.deepseek.com", "model": "deepseek-flash", "protocol": "deepseek"},
    {"id": "qwen", "name": "千问 / 阿里云百炼", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus", "protocol": "qwen"},
    {"id": "zhipu", "name": "智谱 AI", "base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-5.3", "protocol": "zhipu"},
    {"id": "kimi", "name": "Kimi / Moonshot", "base_url": "https://api.moonshot.cn/v1", "model": "kimi-k3", "protocol": "kimi"},
]
PROTOCOLS = ("auto", "openai", "deepseek", "qwen", "zhipu", "kimi")


def reasoning_options(protocol: str, model: str, effort: str | None) -> tuple[dict, str]:
    """Return supported wire parameters and an honest explanation of any mapping.

    DeepSeek: api-docs.deepseek.com/guides/thinking_mode
    GLM: docs.bigmodel.cn/cn/guide/capabilities/thinking
    Kimi: platform.moonshot.cn/docs/guide/use-reasoning-effort
    Qwen: help.aliyun.com/zh/model-studio/deep-thinking
    """
    if effort is None:
        return {}, "沿用模型默认设置"
    if effort not in EFFORTS:
        raise ValueError("推理强度必须是 none / low / medium / high / max")
    name = model.lower().split("/")[-1]
    if protocol == "auto":
        protocol = next((p for prefix, p in (("deepseek", "deepseek"), ("qwen", "qwen"), ("glm", "zhipu"), ("kimi", "kimi")) if name.startswith(prefix)), "openai")
    enabled = effort != "none"
    mapped = "high" if effort == "medium" else effort
    if protocol == "deepseek":
        params = {"thinking": {"type": "enabled" if enabled else "disabled"}}
        if enabled:
            params["reasoning_effort"] = mapped
        return params, "medium 按厂商规则映射为 high" if effort == "medium" else ("关闭思考" if not enabled else f"实际强度：{mapped}")
    if protocol == "qwen":
        if name == "qwen3.7-max-preview" and not enabled:
            # This model's endpoint rejects enable_thinking=False (HTTP 400).
            return {"enable_thinking": True, "thinking_budget": 1024}, "qwen3.7-max-preview 不支持关闭思考；none→low，思考预算：1,024 tokens"
        params = {"enable_thinking": enabled}
        if enabled:
            params["thinking_budget"] = {"low": 1024, "medium": 4096, "high": 16384, "max": 32768}[effort]
        return params, f"思考预算：{params['thinking_budget']:,} tokens（仍受具体模型上限限制）" if enabled else "关闭思考（仅思考模型仍遵循厂商规则）"
    if protocol == "zhipu":
        if name.startswith("glm-5.3"):
            mapped = "low" if not enabled else mapped
            return {"thinking": {"type": "enabled"}, "reasoning_effort": mapped}, f"GLM-5.3 不支持关闭思考；none→low、medium→high，实际：{mapped}"
        if name.startswith("glm-5.2"):
            if effort in ("low", "medium"):
                mapped = "high"
            return {"thinking": {"type": "enabled" if enabled else "disabled"}, **({"reasoning_effort": mapped} if enabled else {})}, f"GLM-5.2 的 low/medium 映射为 high；当前：{mapped}"
        return {"thinking": {"type": "enabled" if enabled else "disabled"}}, "该型号仅适配思考开关，low 至 max 均为开启"
    if protocol == "kimi":
        if name.startswith("kimi-k3"):
            mapped = "low" if not enabled else mapped
            return {"reasoning_effort": mapped}, f"Kimi K3 最低为 low；none→low、medium→high，实际：{mapped}"
        return {"thinking": {"type": "enabled" if enabled else "disabled"}}, "该型号仅适配思考开关，low 至 max 均为开启"
    return ({"reasoning_effort": effort} if enabled else {}), "使用 OpenAI 兼容 reasoning_effort；none 不附加推理参数，具体支持范围由模型决定"


def _crypt(data: bytes, decrypt: bool = False) -> bytes:
    """Windows DPAPI encryption bound to the current user's credentials."""
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_char))]

    buffer = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
    target = Blob()
    dll = ctypes.WinDLL("crypt32", use_last_error=True)
    function = dll.CryptUnprotectData if decrypt else dll.CryptProtectData
    function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    function.restype = wintypes.BOOL
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)):
        raise OSError("无法使用当前 Windows 用户凭据解密已保存的 API Key，请重新填写" if decrypt
                      else "无法使用系统凭据加密保护 API Key")
    try:
        return ctypes.string_at(target.data, target.size)
    finally:
        free = ctypes.WinDLL("kernel32").LocalFree
        free.argtypes = [ctypes.c_void_p]
        free.restype = ctypes.c_void_p
        free(target.data)


def seal(key: str) -> dict:
    if os.name == "nt":
        return {"scheme": "dpapi", "value": base64.b64encode(_crypt(key.encode())).decode()}
    return {"scheme": "local-file", "value": key}


def unseal(value: dict) -> str:
    if not isinstance(value, dict) or not isinstance(value.get("value"), str):
        raise ValueError("已保存的 API Key 格式无效，请重新填写")
    scheme = value.get("scheme")
    if scheme == "dpapi":
        if os.name != "nt":
            raise ValueError("此密钥受 Windows 用户凭据保护，请重新填写")
        try:
            return _crypt(base64.b64decode(value["value"], validate=True), decrypt=True).decode()
        except ValueError as exc:
            raise ValueError("已保存的 API Key 密文格式无效，请重新填写") from exc
    if scheme == "local-file":
        return value["value"]
    raise ValueError("不支持已保存的 API Key 保护格式，请重新填写")


class ProviderStore:
    def __init__(self, path: Path):
        self.path = path
        self.data = {"active": None, "reasoning": "none", "profiles": {}}
        self.warning = ""
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict) or not isinstance(loaded.get("profiles"), dict):
                    raise ValueError("厂商配置结构无效")
                if loaded.get("reasoning", "none") not in EFFORTS:
                    raise ValueError("推理配置无效")
                active = loaded.get("active")
                if active is not None and (not isinstance(active, str) or active not in loaded["profiles"]):
                    raise ValueError("当前厂商配置无效")
                for key, profile in loaded["profiles"].items():
                    if not isinstance(profile, dict) or profile.get("id") != key:
                        raise ValueError("厂商配置条目无效")
                    if any(not isinstance(profile.get(field), str) for field in ("name", "base_url", "model", "protocol")) or not isinstance(profile.get("api_key"), dict):
                        raise ValueError("厂商配置条目缺少必要字段")
                self.data.update(loaded)
            except ValueError:
                backup = path.with_name(path.name + '.corrupt-' + uuid.uuid4().hex + '.bak')
                try:
                    path.rename(backup)
                except OSError as exc:
                    raise OSError("厂商配置损坏且无法备份，已停止加载并保留原文件") from exc
                self.warning = f"厂商配置损坏，已重置。原文件已备份至：{backup}"
            except OSError as exc:
                raise OSError("无法读取厂商配置，已保留原文件，请检查文件权限或磁盘状态") from exc

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=".providers-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
            self.data = data
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def public(self) -> dict:
        profiles = []
        for profile in self.data["profiles"].values():
            profiles.append({k: v for k, v in profile.items() if k != "api_key"} | {"has_key": bool(profile.get("api_key"))})
        return {"profiles": profiles, "presets": PRESETS, "active": self.data["active"], "reasoning": self.data["reasoning"], "warning": self.warning}

    def prepare(self, payload: dict) -> tuple[dict, str]:
        profile_id = str(payload.get("id") or "")
        if profile_id not in self.data["profiles"] and profile_id not in {p["id"] for p in PRESETS}:
            profile_id = uuid.uuid4().hex
        previous = self.data["profiles"].get(profile_id, {})
        url = str(payload.get("base_url") or "").strip().rstrip("/")
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError
            _ = parsed.port  # Reject malformed and out-of-range ports before saving.
        except ValueError as exc:
            raise ValueError("Base URL 需要完整 http(s) 地址和有效端口，不能包含账号、密码、查询参数或片段") from exc
        model = str(payload.get("model") or "").strip()
        protocol = str(payload.get("protocol") or "openai")
        if not model or len(model) > 200 or any(c.isspace() for c in model):
            raise ValueError("请填写有效的模型 ID")
        if protocol not in PROTOCOLS:
            raise ValueError("未知厂商协议")
        key = str(payload.get("api_key") or "").strip()
        if not key:
            if previous.get("base_url") != url:
                raise ValueError("新增厂商或修改地址后，请填写对应的 API Key")
            if not previous.get("api_key"):
                raise ValueError("请填写 API Key")
            key = unseal(previous["api_key"])
        if any(ord(c) < 33 or ord(c) > 126 for c in key):
            raise ValueError("API Key 不能包含空白、换行或非 ASCII 字符")
        profile = {"id": profile_id, "name": str(payload.get("name") or model).strip()[:80],
                   "base_url": url, "model": model, "protocol": protocol, "api_key": seal(key)}
        return profile, key

    def save(self, profile: dict) -> None:
        data = {**self.data, "profiles": {**self.data["profiles"], profile["id"]: profile}, "active": profile["id"]}
        self._write(data)

    def active(self) -> tuple[dict, str] | None:
        profile = self.data["profiles"].get(self.data["active"])
        return (profile, unseal(profile["api_key"])) if profile else None

    def set_model(self, model: str) -> None:
        active = self.data["active"]
        if active in self.data["profiles"]:
            updated = {**self.data["profiles"][active], "model": model}
            self._write({**self.data, "profiles": {**self.data["profiles"], active: updated}})

    def set_effort(self, effort: str) -> None:
        if effort not in EFFORTS:
            raise ValueError("未知推理强度")
        self._write({**self.data, "reasoning": effort})
