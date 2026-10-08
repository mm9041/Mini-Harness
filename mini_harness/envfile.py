"""零依赖 ``.env`` 读取。

为什么不直接用 python-dotenv:整个项目刻意做到零第三方依赖,不想为了几行配置
破例。支持的语法覆盖了日常用法:

    KEY=value
    export KEY=value
    KEY="value with spaces"
    KEY='value'
    KEY=value   # 行尾注释
    KEY="value" # 行尾注释(注释先剥,引号后剥)
    # 整行注释

不支持(刻意不实现):多行值、``${VAR}`` 插值、``\\n`` 转义。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping

__all__ = ["parse_env_text", "load_env_file", "discover_env_file"]


def parse_env_text(text: str) -> dict[str, str]:
    """把 ``.env`` 文本解析成字典。遇到看不懂的行就跳过,不抛异常。"""
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        values[key] = _clean_value(value)
    return values


def _is_quoted(value: str) -> bool:
    """整段被同一对引号包住。"""
    return len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'"


def _clean_value(value: str) -> str:
    """剥掉行尾注释,再剥掉包裹引号 —— 两步都能发生,顺序不能反。

    `KEY="v" # 注释` 这种写法很常见,但它的首尾字符并不相同(尾是注释的最后一个
    字符),所以不能只判一次引号:先剥注释,剩下 `"v"` 才是完整的引号值。
    """
    value = value.strip()
    if _is_quoted(value):
        return value[1:-1]  # 加引号:原样保留,不做注释剥离

    marker = value.find(" #")  # 未加引号时,` #` 之后算注释
    if marker != -1:
        value = value[:marker].rstrip()
        if _is_quoted(value):  # 剥完注释才显形的引号值
            return value[1:-1]
    return value.rstrip()


def load_env_file(
    path: str | Path,
    env: MutableMapping[str, str] | None = None,
    override: bool = False,
) -> dict[str, str]:
    """读取 ``.env`` 并写入环境变量,返回**实际生效**的那些键值。

    默认不覆盖已存在的环境变量 —— 这样"在 shell 里临时 export 一个值"总能盖住
    ``.env``,符合直觉。
    """
    target = Path(path)
    if not target.is_file():
        return {}

    parsed = parse_env_text(target.read_text(encoding="utf-8-sig"))
    destination = os.environ if env is None else env
    applied: dict[str, str] = {}
    for key, value in parsed.items():
        if override or key not in destination:
            destination[key] = value
            applied[key] = value
    return applied


def discover_env_file(
    start: str | Path | None = None,
    env: MutableMapping[str, str] | None = None,
) -> Path | None:
    """按优先级找 ``.env``:显式指定 > 起始目录 > 包所在的项目根目录。

    显式指定来自 ``MINI_HARNESS_ENV_FILE``;把它指向一个不存在的路径,
    就等于关掉 ``.env`` 自动读取。
    """
    source = os.environ if env is None else env
    explicit = (source.get("MINI_HARNESS_ENV_FILE") or "").strip()
    if explicit:
        candidate = Path(explicit).expanduser()
        return candidate if candidate.is_file() else None

    candidates = [
        Path(start or Path.cwd()) / ".env",
        Path(__file__).resolve().parent.parent / ".env",  # 项目根目录
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
