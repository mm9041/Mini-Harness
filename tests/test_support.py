"""``tests/support.py`` 的测试 —— 重点是那个 Windows 目录锁的确定性复现。

这个文件存在的理由:用户在 Windows 上跑测试时,有 3 个测试的**断言全部通过**,却因为
``asyncTearDown`` 里删临时目录撞上 ``WinError 32`` 被判为 error。偶发、依赖负载,
本机跑不出来。所以这里不靠"多跑几次碰运气",而是**构造出那个状态**:
让一个活着的子进程把临时目录当成自己的工作目录 —— 这在 Windows 上必然让
``shutil.rmtree`` 失败,然后验证重试版的 ``remove_tree`` 能等到子进程退出后删掉它。
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import unittest

from .support import make_temp_dir, remove_tree


class RemoveTreeTests(unittest.IsolatedAsyncioTestCase):
    async def test_removes_a_plain_directory(self) -> None:
        target = make_temp_dir()
        (target / "a.txt").write_text("x", encoding="utf-8")
        (target / "sub").mkdir()

        self.assertTrue(await remove_tree(target))
        self.assertFalse(target.exists())

    async def test_missing_directory_is_not_an_error(self) -> None:
        self.assertTrue(await remove_tree(make_temp_dir()))

    async def test_second_removal_is_a_no_op(self) -> None:
        target = make_temp_dir()
        self.assertTrue(await remove_tree(target))
        self.assertTrue(await remove_tree(target))


@unittest.skipUnless(
    os.name == "nt",
    "WinError 32(删除他人工作目录被拒)只在 Windows 上出现;POSIX 允许删除在用目录",
)
class WindowsDirectoryLockTests(unittest.IsolatedAsyncioTestCase):
    async def test_remove_tree_waits_out_a_child_holding_the_directory(self) -> None:
        workdir = make_temp_dir(prefix="mini-harness-lock-")

        # 一个活着的子进程把 workdir 当自己的 cwd。用 sys.executable 而不是 shell,
        # 保证跟平台/已装 shell 无关,一定起得来。
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(1.5)",
            cwd=str(workdir),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        try:
            # 这里必须等一小段:**实测子进程要几十毫秒才真正把 workdir 当成 cwd** ——
            # 刚 spawn 完就 rmtree 是能成功的(还没锁上),50ms 之后必定 WinError 32。
            # 不等的话,这条测试本身就变成在赌时序(踩过:偶发失败)。
            await asyncio.sleep(0.3)

            # ① 先证明这个状态确实会让"直接删"失败 —— 这正是用户看到的报错。
            with self.assertRaises(PermissionError):
                shutil.rmtree(workdir)

            # ② 重试版应当跨过这个窗口(子进程退出后),成功删掉而不是抛错。
            self.assertTrue(
                await remove_tree(workdir, attempts=40, delay=0.05),
                "remove_tree 应当在子进程退出后删掉目录",
            )
            self.assertFalse(workdir.exists())
        finally:
            if child.returncode is None:
                child.kill()
                await child.wait()
            await remove_tree(workdir)


if __name__ == "__main__":
    unittest.main()
