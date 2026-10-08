"""Install the pinned DSH runner outside editable workspaces. Run with Python."""
import os
from pathlib import Path
import shutil
import subprocess

source = Path(__file__).resolve().parent.parent
target = Path(os.environ.get("MINI_HARNESS_SANDBOX_RUNTIME_ROOT") or
              Path.home() / ".mini-harness" / "sandbox-runtime").resolve()
if target == source or source in target.parents:
    raise SystemExit("沙箱运行时必须安装在 mini-harness 工作区之外")
npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
if npm is None:
    raise SystemExit("请先安装 Node.js（含 npm）")
target.mkdir(parents=True, exist_ok=True)
for name in ("package.json", "package-lock.json"):
    shutil.copy2(source / name, target / name)
shutil.copy2(source / "runtime/sandbox_runner.cjs", target / "sandbox_runner.cjs")
subprocess.run([npm, "ci", "--no-fund", "--no-audit"], cwd=target, check=True)
print(f"Sandbox runtime: {target}")
