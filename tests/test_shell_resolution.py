"""shell 选择逻辑的测试。

重点是那条**静默错位**:Windows 的 ``WindowsApps\\bash.EXE`` 是 0 字节应用执行别名,
实际指向 WSL —— 选中它,所有命令就跑进 Linux 子系统了。这个文件把它钉死。

``_resolve_windows_shell`` 内部不判断 ``os.name``,所以换掉两个探针函数就能在
任何平台上测;``is_windows_app_alias`` 用 ``ntpath`` 解析,同样与平台无关。
"""

from __future__ import annotations

import contextlib
import ntpath
import os
import unittest
from pathlib import Path
from unittest import mock

from mini_harness.builtin_tools import shell as shell_module
from mini_harness.tools import ToolCallContext

WSL_ALIAS = ntpath.join(
    r"C:\Users\someone\AppData\Local\Microsoft\WindowsApps", "bash.EXE"
)
GIT_BASH = r"C:\Program Files\Git\usr\bin\bash.exe"
GIT_BIN_BASH = r"C:\Program Files\Git\bin\bash.exe"
GIT_USR_BIN = r"C:\Program Files\Git\usr\bin"
GIT_EXE = r"C:\Program Files\Git\cmd\git.exe"
PORTABLE_BASH = r"C:\tools\PortableGit\usr\bin\bash.exe"
PWSH = r"C:\Program Files\PowerShell\7\pwsh.exe"
POWERSHELL = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
CMD = r"C:\WINDOWS\system32\cmd.exe"

PWSH_ARGV = [PWSH, "-NoProfile", "-NonInteractive", "-Command"]
POWERSHELL_ARGV = [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command"]


def _norm(path: str) -> str:
    return ntpath.normcase(ntpath.normpath(path))


class FakeMachine:
    """替身机器:声明 PATH 上有什么、磁盘上有什么、各自多大。"""

    def __init__(
        self,
        which: dict[str, str] | None = None,
        existing: set[str] | None = None,
        sizes: dict[str, int] | None = None,
    ) -> None:
        self.which = which or {}
        self.existing = {_norm(path) for path in (existing or set())}
        self.sizes = {_norm(k): v for k, v in (sizes or {}).items()}

    def find_on_path(self, name: str) -> str | None:
        return self.which.get(name)

    def file_exists(self, path: str) -> bool:
        return _norm(path) in self.existing

    def file_size(self, path: str) -> int:
        return self.sizes.get(_norm(path), -1)

    @contextlib.contextmanager
    def patched(self):
        with mock.patch.object(shell_module, "_find_on_path", self.find_on_path), \
                mock.patch.object(shell_module, "_file_exists", self.file_exists), \
                mock.patch.object(shell_module, "_file_size", self.file_size):
            yield self


class WindowsAppAliasTests(unittest.TestCase):
    def test_zero_byte_windowsapps_entry_is_an_alias(self) -> None:
        with FakeMachine(sizes={WSL_ALIAS: 0}).patched():
            self.assertTrue(shell_module.is_windows_app_alias(WSL_ALIAS))

    def test_real_binary_under_windowsapps_is_not_an_alias(self) -> None:
        """目录对但体积不为 0 —— 不是别名,不能误杀。"""
        with FakeMachine(sizes={WSL_ALIAS: 2_311_528}).patched():
            self.assertFalse(shell_module.is_windows_app_alias(WSL_ALIAS))

    def test_git_bash_is_not_an_alias(self) -> None:
        with FakeMachine(sizes={GIT_BASH: 2_311_528}).patched():
            self.assertFalse(shell_module.is_windows_app_alias(GIT_BASH))

    def test_missing_path_is_not_an_alias(self) -> None:
        with FakeMachine().patched():
            self.assertFalse(shell_module.is_windows_app_alias(WSL_ALIAS))


class WindowsShellResolutionTests(unittest.TestCase):
    def test_skips_wsl_alias_and_uses_the_bash_shipped_with_git(self) -> None:
        """本机的真实情况:PATH 上的 bash 是 WSL 别名,git 在 PATH 上。"""
        machine = FakeMachine(
            which={"bash": WSL_ALIAS, "git": GIT_EXE},
            existing={GIT_BASH},
            sizes={WSL_ALIAS: 0, GIT_BASH: 2_311_528},
        )
        with machine.patched():
            self.assertEqual(
                shell_module._resolve_windows_shell(), ("bash", [GIT_BASH, "-c"])
            )

    def test_uses_native_bash_on_path_when_it_is_real(self) -> None:
        machine = FakeMachine(
            which={"bash": PORTABLE_BASH},
            existing={PORTABLE_BASH},
            sizes={PORTABLE_BASH: 1_000_000},
        )
        with machine.patched():
            self.assertEqual(
                shell_module._resolve_windows_shell(), ("bash", [PORTABLE_BASH, "-c"])
            )

    def test_falls_back_to_well_known_path_when_git_is_absent(self) -> None:
        machine = FakeMachine(existing={GIT_BASH}, sizes={GIT_BASH: 2_311_528})
        with machine.patched():
            self.assertEqual(
                shell_module._resolve_windows_shell(), ("bash", [GIT_BASH, "-c"])
            )

    def test_falls_back_to_pwsh_when_there_is_no_bash_at_all(self) -> None:
        machine = FakeMachine(
            which={"pwsh": PWSH, "powershell": POWERSHELL, "cmd": CMD}
        )
        with machine.patched():
            self.assertEqual(
                shell_module._resolve_windows_shell(), ("pwsh", PWSH_ARGV)
            )

    def test_falls_back_to_cmd_as_the_last_resort(self) -> None:
        machine = FakeMachine(which={"cmd": CMD})
        with machine.patched():
            self.assertEqual(
                shell_module._resolve_windows_shell(), ("cmd", [CMD, "/c"])
            )

    def test_returns_none_when_nothing_is_available(self) -> None:
        with FakeMachine().patched():
            self.assertIsNone(shell_module._resolve_windows_shell())


class ExplicitOverrideTests(unittest.TestCase):
    """显式指定永远优先 —— 包括用户就是想要 WSL 别名的情况。"""

    def test_explicit_paths_map_to_the_right_dialect(self) -> None:
        cases = {
            PWSH: ("pwsh", PWSH_ARGV),
            POWERSHELL: ("powershell", POWERSHELL_ARGV),
            CMD: ("cmd", [CMD, "/c"]),
            GIT_BASH: ("bash", [GIT_BASH, "-c"]),
        }
        for prefer, expected in cases.items():
            with self.subTest(prefer=prefer):
                self.assertEqual(shell_module.detect_shell(prefer), expected)

    def test_explicit_alias_is_still_honoured(self) -> None:
        machine = FakeMachine(sizes={WSL_ALIAS: 0})
        with machine.patched():
            self.assertEqual(
                shell_module.detect_shell(WSL_ALIAS), ("bash", [WSL_ALIAS, "-c"])
            )


class GitBashEntryPointTests(unittest.TestCase):
    """`bin\\bash.exe` 与 `usr\\bin\\bash.exe` 不一样 —— 实测前者才会带上 coreutils。

    `usr\\bin\\bash.exe` 原样继承 PATH,`grep`/`sed`/`wc`/`uname` 全都找不到,
    `find` 更会落到 Windows 自带的 `find.exe`(完全不同的程序)。所以必须优先 `bin`。
    """

    def test_prefers_bin_over_usr_bin(self) -> None:
        bin_bash, usr_bash = GIT_BIN_BASH, GIT_BASH
        machine = FakeMachine(
            which={"git": GIT_EXE},
            existing={bin_bash, usr_bash},
            sizes={bin_bash: 45_560, usr_bash: 2_311_528},
        )
        with machine.patched():
            self.assertEqual(shell_module._bash_shipped_with_git(), bin_bash)

    def test_falls_back_to_usr_bin_when_bin_is_absent(self) -> None:
        machine = FakeMachine(
            which={"git": GIT_EXE}, existing={GIT_BASH}, sizes={GIT_BASH: 2_311_528}
        )
        with machine.patched():
            self.assertEqual(shell_module._bash_shipped_with_git(), GIT_BASH)

    def test_root_is_derived_from_both_entry_points(self) -> None:
        expected = [
            ntpath.normcase(r"C:\Program Files\Git\usr\bin"),
            ntpath.normcase(r"C:\Program Files\Git\mingw64\bin"),
        ]
        for entry in (GIT_BIN_BASH, GIT_BASH):
            with self.subTest(entry=entry):
                self.assertEqual(
                    [ntpath.normcase(d) for d in shell_module._coreutils_dirs(entry)],
                    expected,
                )

    def test_child_environment_prepends_missing_coreutils(self) -> None:
        fake_dirs = {GIT_USR_BIN: True}
        with mock.patch.object(shell_module.os.path, "isdir", fake_dirs.get), \
                mock.patch.dict(shell_module.os.environ, {"PATH": r"C:\WINDOWS\system32"}, clear=False):
            env = shell_module._child_environment("bash", GIT_BASH)

        self.assertIsNotNone(env)
        parts = env["PATH"].split(os.pathsep)
        self.assertEqual(ntpath.normcase(parts[0]), ntpath.normcase(GIT_USR_BIN))
        self.assertIn(r"C:\WINDOWS\system32", parts)

    def test_child_environment_is_none_when_nothing_to_add(self) -> None:
        with mock.patch.object(shell_module.os.path, "isdir", lambda _p: False):
            self.assertIsNone(shell_module._child_environment("bash", GIT_BASH))

    def test_child_environment_is_none_for_non_bash(self) -> None:
        self.assertIsNone(shell_module._child_environment("pwsh", PWSH))


class BashCoreutilsIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """真的起一个子进程,验证"选了 bash 就真有一套 coreutils"。"""

    async def test_chosen_shell_can_see_coreutils(self) -> None:
        label, _argv = shell_module.detect_shell()
        if label != "bash":
            self.skipTest(f"本机自动选中的是 {label},没有 coreutils 语义可验证")

        tool = shell_module.make_shell_tool(default_cwd=os.getcwd())
        context = ToolCallContext(call_id="coreutils", cwd=Path(os.getcwd()))
        result = await tool.handler(
            {
                "command": (
                    "command -v grep >/dev/null && command -v sed >/dev/null "
                    "&& command -v wc >/dev/null && command -v uname >/dev/null "
                    "&& echo ALL_COREUTILS_OK || echo MISSING_COREUTILS"
                )
            },
            context,
        )

        self.assertIn("ALL_COREUTILS_OK", result.content, result.content)


if __name__ == "__main__":
    unittest.main()
