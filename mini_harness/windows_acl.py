"""Prepare caller-owned workspaces for DSH without taking ownership or granting FullControl."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import threading
import tempfile
import uuid
from datetime import datetime, timezone


READ_CONTROL = 0x20000
WRITE_DAC = 0x40000
WRITE_OWNER = 0x80000
MODIFY = 0x1301BF
INHERIT = 0x03  # OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE
# Serializes this process only. Identical repairs of an unchanged ACL are
# idempotent, but read/merge/write is not atomic across processes: an external
# ACL edit can be overwritten. This lock is not cross-process coordination.
_lock = threading.RLock()


class _Trustee(ctypes.Structure):
    _fields_ = [("multiple", ctypes.c_void_p), ("operation", ctypes.c_int),
                ("form", ctypes.c_int), ("type", ctypes.c_int), ("name", ctypes.c_void_p)]


class _ExplicitAccess(ctypes.Structure):
    _fields_ = [("permissions", wintypes.DWORD), ("mode", ctypes.c_int),
                ("inheritance", wintypes.DWORD), ("trustee", _Trustee)]


class _AclSize(ctypes.Structure):
    _fields_ = [("count", wintypes.DWORD), ("used", wintypes.DWORD), ("free", wintypes.DWORD)]


class _AllowedAce(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ubyte), ("flags", ctypes.c_ubyte),
                ("size", wintypes.WORD), ("mask", wintypes.DWORD)]


class _WindowsAcl:
    def __init__(self):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        advapi = ctypes.WinDLL("advapi32", use_last_error=True)
        ptr = ctypes.c_void_p
        out = ctypes.POINTER(ptr)

        def bind(dll, name, result, *args):
            fn = getattr(dll, name)
            fn.restype, fn.argtypes = result, list(args)
            return fn

        self.open = bind(kernel, "CreateFileW", wintypes.HANDLE, wintypes.LPCWSTR,
                         wintypes.DWORD, wintypes.DWORD, ptr, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
        self.close = bind(kernel, "CloseHandle", wintypes.BOOL, wintypes.HANDLE)
        self.free = bind(kernel, "LocalFree", ptr, ptr)
        self.process = bind(kernel, "GetCurrentProcess", wintypes.HANDLE)
        self.open_token = bind(advapi, "OpenProcessToken", wintypes.BOOL,
                               wintypes.HANDLE, wintypes.DWORD, out)
        self.token_info = bind(advapi, "GetTokenInformation", wintypes.BOOL,
                               wintypes.HANDLE, ctypes.c_int, ptr, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD))
        self.get_security = bind(advapi, "GetSecurityInfo", wintypes.DWORD,
                                 wintypes.HANDLE, ctypes.c_int, wintypes.DWORD, out, out, out, out, out)
        self.set_security = bind(advapi, "SetSecurityInfo", wintypes.DWORD,
                                 wintypes.HANDLE, ctypes.c_int, wintypes.DWORD, ptr, ptr, ptr, ptr)
        self.to_sddl = bind(advapi, "ConvertSecurityDescriptorToStringSecurityDescriptorW",
                            wintypes.BOOL, ptr, wintypes.DWORD, wintypes.DWORD, out, ptr)
        self.to_sid = bind(advapi, "ConvertSidToStringSidW", wintypes.BOOL, ptr, out)
        self.equal = bind(advapi, "EqualSid", wintypes.BOOL, ptr, ptr)
        self.acl_info = bind(advapi, "GetAclInformation", wintypes.BOOL, ptr, ptr, wintypes.DWORD, ctypes.c_int)
        self.get_ace = bind(advapi, "GetAce", wintypes.BOOL, ptr, wintypes.DWORD, out)
        self.merge = bind(advapi, "SetEntriesInAclW", wintypes.DWORD,
                          wintypes.ULONG, ctypes.POINTER(_ExplicitAccess), ptr, out)

    @staticmethod
    def check(ok):
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())

    @staticmethod
    def check_code(code):
        if code:
            raise ctypes.WinError(code)

    def handle(self, path, access):
        # Keep the directory from being renamed/replaced while inspecting and repairing it.
        result = self.open(str(path), access, 3, None, 3, 0x02000000, None)
        if result == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        return result

    def can_access(self, path, access):
        try:
            handle = self.handle(path, access)
        except OSError as exc:
            if exc.winerror == 5:
                return False
            raise
        self.close(handle)
        return True

    def current_user(self):
        token = ctypes.c_void_p()
        self.check(self.open_token(self.process(), 0x08, ctypes.byref(token)))
        try:
            size = wintypes.DWORD()
            self.token_info(token, 1, None, 0, ctypes.byref(size))
            if not size.value:
                raise ctypes.WinError(ctypes.get_last_error())
            data = ctypes.create_string_buffer(size.value)
            self.check(self.token_info(token, 1, data, size, ctypes.byref(size)))
            # TOKEN_USER starts with SID_AND_ATTRIBUTES. Keep its owning buffer alive.
            return data, ctypes.c_void_p.from_buffer(data).value
        finally:
            self.close(token)

    def string(self, convert, *args):
        result = ctypes.c_void_p()
        self.check(convert(*args, ctypes.byref(result)))
        try:
            return ctypes.wstring_at(result)
        finally:
            self.free(result)

    def has_inheritable(self, acl, sid, rights):
        if not acl:
            # Do not convert a NULL DACL (Everyone FullControl) into a restrictive ACL.
            raise RuntimeError("目录使用 NULL DACL，不能自动准备沙箱权限")
        size = _AclSize()
        self.check(self.acl_info(acl, ctypes.byref(size), ctypes.sizeof(size), 2))
        mask = 0
        for index in range(size.count):
            pointer = ctypes.c_void_p()
            self.check(self.get_ace(acl, index, ctypes.byref(pointer)))
            ace = _AllowedAce.from_address(pointer.value)
            if (ace.type == 0 and ace.flags & INHERIT == INHERIT
                    and not ace.flags & 0x0C and self.equal(pointer.value + 8, sid)):
                mask |= ace.mask
        return mask & rights == rights

    def prepare(self, path, backup_root):
        handle = self.handle(path, READ_CONTROL)
        descriptor = ctypes.c_void_p()
        try:
            owner, acl = ctypes.c_void_p(), ctypes.c_void_p()
            self.check_code(self.get_security(handle, 1, 7, ctypes.byref(owner), None,
                                              ctypes.byref(acl), None, ctypes.byref(descriptor)))
            user_buffer, sid = self.current_user()
            modify = self.has_inheritable(acl, sid, MODIFY)
            inherit_owner = self.has_inheritable(acl, sid, WRITE_OWNER)
            write_owner = self.can_access(path, WRITE_OWNER)
            if not self.can_access(path, WRITE_DAC):
                raise RuntimeError("当前用户无权修改目录 DACL (WRITE_DAC)")
            if modify and inherit_owner and write_owner:
                return None
            if inherit_owner and not write_owner:
                raise RuntimeError("WRITE_OWNER 已授权但仍被拒绝；请检查显式拒绝规则")
            if not self.equal(owner, sid):
                raise RuntimeError("目录不属于当前用户，不能自动修改 ACL；请由属主配置当前用户的可继承 Modify/WRITE_OWNER 权限")
            # Open without privileges/ownership changes; an explicit deny remains authoritative.
            writable = self.handle(path, WRITE_DAC | READ_CONTROL)
            try:
                sddl_ptr = ctypes.c_void_p()
                self.check(self.to_sddl(descriptor, 1, 7, ctypes.byref(sddl_ptr), None))
                try:
                    sddl = ctypes.wstring_at(sddl_ptr)
                finally:
                    self.free(sddl_ptr)
                additions = []
                if not modify:
                    # DSH removes Authenticated Users/Administrators from its write token.
                    # An inheritable user-SID Modify grant survives that filtering.
                    additions.append((MODIFY, INHERIT, "Modify (this folder, subfolders and files)"))
                if not inherit_owner:
                    # DSH propagates the Low label to existing descendants in the
                    # same native operation. Those objects also need WRITE_OWNER.
                    additions.append((WRITE_OWNER, INHERIT, "WRITE_OWNER (this folder, subfolders and files)"))
                backup_root = Path(backup_root).resolve()
                if backup_root == path or path in backup_root.parents:
                    raise RuntimeError("ACL 备份目录必须位于工作区之外")
                backup_root.mkdir(parents=True, exist_ok=True)
                record = {"version": 1, "workspace": str(path), "user_sid": self.string(self.to_sid, sid),
                          "created_at": datetime.now(timezone.utc).isoformat(), "sddl_before": sddl,
                          "added_rights": [item[2] for item in additions]}
                backup = backup_root / f"{uuid.uuid4().hex}.json"
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                                     dir=backup_root, prefix=".acl-", suffix=".tmp",
                                                     delete=False) as stream:
                        temporary = Path(stream.name)
                        json.dump(record, stream, ensure_ascii=False, indent=2)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, backup)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
                entries = (_ExplicitAccess * len(additions))()
                for entry, (rights, inheritance, _) in zip(entries, additions):
                    entry.permissions, entry.mode, entry.inheritance = rights, 1, inheritance
                    entry.trustee = _Trustee(None, 0, 0, 1, sid)
                merged = ctypes.c_void_p()
                self.check_code(self.merge(len(entries), entries, acl, ctypes.byref(merged)))
                try:
                    self.check_code(self.set_security(writable, 1, 4, None, None, merged, None))
                finally:
                    self.free(merged)
                if not self.can_access(path, WRITE_OWNER):
                    raise RuntimeError(f"ACL 已备份到 {backup}，但 WRITE_OWNER 仍被拒绝；请检查显式拒绝规则")
                return {"workspace": str(path), "backup_path": str(backup), "added_rights": record["added_rights"]}
            finally:
                self.close(writable)
        finally:
            if descriptor:
                self.free(descriptor)
            self.close(handle)


def prepare_workspace_acl(workspace, backup_root):
    """Prepare only the selected workspace, without ownership change or escalation.

    The original SDDL backup is for diagnosis and manual recovery; there is no
    automatic rollback. A complete backup is published before any ACL write.
    Caught write failures remove temporary files; a process crash can leave a
    .tmp file, but never exposes that partial file as a completed JSON backup.
    Coordination is process-local; concurrent external ACL edits are not guarded.
    """
    path = Path(workspace).resolve()
    try:
        with _lock:
            return _WindowsAcl().prepare(path, backup_root)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"SANDBOX_UNAVAILABLE: 工作区 ACL 初始化失败：{path}；{exc}。"
            "DSH 需要当前用户的可继承 Modify/WRITE_OWNER 及目录 WRITE_DAC 权限。"
            "请先处理目录权限；重复执行命令或申请完全访问不会修复沙箱。"
        ) from exc
