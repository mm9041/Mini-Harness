"""Open a folder chooser on the computer running the local harness."""
from __future__ import annotations

import base64
import os
import subprocess
import sys


def choose_directory(initial: str) -> str | None:
    if os.name == "nt":
        # Pass the initial path as data, never interpolate it into PowerShell code.
        script = r"""
Add-Type -AssemblyName System.Windows.Forms
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'Select workspace folder'
$dialog.SelectedPath = $env:MINI_HARNESS_PICKER_INITIAL
$dialog.ShowNewFolderButton = $true
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.ShowInTaskbar = $false
$owner.Opacity = 0
try {
    $owner.Show()
    $owner.Activate()
    if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
        [Console]::Write([Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($dialog.SelectedPath)))
    }
} finally {
    $dialog.Dispose()
    $owner.Dispose()
}
"""
        env = dict(os.environ, MINI_HARNESS_PICKER_INITIAL=initial)
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-STA", "-EncodedCommand",
             base64.b64encode(script.encode("utf-16-le")).decode("ascii")],
            capture_output=True, env=env, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode:
            raise RuntimeError("无法打开本地文件夹选择窗口")
        encoded = result.stdout.strip()
        return base64.b64decode(encoded, validate=True).decode("utf-8") if encoded else None

    # Tk must run in its own main thread, not an HTTP worker thread.
    script = """
import tkinter as tk
from tkinter import filedialog
import base64, sys
root = tk.Tk()
root.withdraw()
try:
    path = filedialog.askdirectory(parent=root, title='Select workspace folder', initialdir=sys.argv[1], mustexist=True)
    print(base64.b64encode(path.encode('utf-8')).decode('ascii'), end='')
finally:
    root.destroy()
"""
    result = subprocess.run([sys.executable, "-c", script, initial], capture_output=True)
    if result.returncode:
        raise RuntimeError("无法打开本地文件夹选择窗口；此系统需要 Tk 桌面支持")
    return base64.b64decode(result.stdout.strip()).decode("utf-8") or None
