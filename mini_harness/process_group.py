"""Windows jobs own shell descendants before the initial thread is resumed.

Popen supports CREATE_SUSPENDED but closes the primary thread handle. Toolhelp
reopens that still-suspended thread; ambiguous enumeration fails closed. As with
DSH, a host killed between creation and assignment may leave a suspended child.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes

CREATE_SUSPENDED = 0x00000004


class _ThreadEntry(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD),
                ("thread_id", wintypes.DWORD), ("owner_pid", wintypes.DWORD),
                ("base_priority", wintypes.LONG), ("delta_priority", wintypes.LONG),
                ("flags", wintypes.DWORD)]


class WindowsProcessGroup:
    def __init__(self):
        size_t = ctypes.c_size_t

        class Limits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                        ("flags", wintypes.DWORD), ("min_ws", size_t), ("max_ws", size_t),
                        ("active", wintypes.DWORD), ("affinity", size_t),
                        ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

        class Extended(ctypes.Structure):
            _fields_ = [("basic", Limits), ("io", IO), ("process_memory", size_t),
                        ("job_memory", size_t), ("peak_process", size_t), ("peak_job", size_t)]

        self.dll = ctypes.WinDLL("kernel32", use_last_error=True)
        self.dll.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.dll.CreateJobObjectW.restype = wintypes.HANDLE
        self.dll.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.dll.SetInformationJobObject.restype = wintypes.BOOL
        self.dll.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.dll.AssignProcessToJobObject.restype = wintypes.BOOL
        self.dll.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.dll.TerminateJobObject.restype = wintypes.BOOL
        self.dll.CloseHandle.argtypes = [wintypes.HANDLE]
        self.dll.CloseHandle.restype = wintypes.BOOL
        self.dll.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        self.dll.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        for name in ("Thread32First", "Thread32Next"):
            method = getattr(self.dll, name)
            method.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ThreadEntry)]
            method.restype = wintypes.BOOL
        self.dll.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.dll.OpenThread.restype = wintypes.HANDLE
        self.dll.GetProcessIdOfThread.argtypes = [wintypes.HANDLE]
        self.dll.GetProcessIdOfThread.restype = wintypes.DWORD
        self.dll.ResumeThread.argtypes = [wintypes.HANDLE]
        self.dll.ResumeThread.restype = wintypes.DWORD
        self.handle = self.dll.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.dll.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise OSError("无法设置作业进程树清理策略")

    def assign(self, process) -> None:
        if not self.dll.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def assign_suspended(self, process) -> None:
        """Assign a freshly CREATE_SUSPENDED child, then permit its code to run.

        The caller must kill/wait the process and close this group on failure.
        This method must not be used with an already running process.
        """
        self.assign(process)
        self._resume_initial_thread(process)

    def _resume_initial_thread(self, process) -> None:
        snapshot = self.dll.CreateToolhelp32Snapshot(0x4, 0)  # TH32CS_SNAPTHREAD
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        thread_ids = []
        try:
            entry = _ThreadEntry()
            entry.size = ctypes.sizeof(entry)
            found = self.dll.Thread32First(snapshot, ctypes.byref(entry))
            while found:
                if entry.owner_pid == process.pid:
                    thread_ids.append(entry.thread_id)
                entry.size = ctypes.sizeof(entry)
                found = self.dll.Thread32Next(snapshot, ctypes.byref(entry))
            error = ctypes.get_last_error()
            if error != 18:  # ERROR_NO_MORE_FILES is the expected end of enumeration.
                raise ctypes.WinError(error)
        finally:
            self.dll.CloseHandle(snapshot)
        if len(thread_ids) != 1:
            raise OSError("无法确认挂起进程的唯一初始线程，拒绝恢复执行")
        # THREAD_SUSPEND_RESUME | THREAD_QUERY_LIMITED_INFORMATION
        thread = self.dll.OpenThread(0x0002 | 0x0800, False, thread_ids[0])
        if not thread:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if self.dll.GetProcessIdOfThread(thread) != process.pid:
                raise OSError("挂起线程不属于目标进程，拒绝恢复执行")
            previous = self.dll.ResumeThread(thread)
            if previous == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
            if previous != 1:
                raise OSError(f"初始线程挂起计数异常: {previous}")
        finally:
            self.dll.CloseHandle(thread)

    def terminate(self) -> None:
        if self.handle:
            self.dll.TerminateJobObject(self.handle, 1)

    def close(self) -> None:
        if self.handle:
            self.dll.CloseHandle(self.handle)
            self.handle = None
