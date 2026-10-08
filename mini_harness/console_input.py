"""Cancellable TTY input without a lingering input() worker thread."""
from __future__ import annotations

import asyncio
import os
import sys


async def read_line(prompt: str) -> str:
    sys.stdout.write(prompt)
    sys.stdout.flush()
    if os.name != "nt":
        import select
        import termios
        try:
            while not select.select([sys.stdin], [], [], 0)[0]:
                await asyncio.sleep(0.05)
            line = sys.stdin.readline()
            if not line:
                raise EOFError
            return line.rstrip("\r\n")
        except asyncio.CancelledError:
            # Discard the unfinished question input before returning to the REPL.
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
            sys.stdout.write("\n")
            sys.stdout.flush()
            raise

    import msvcrt
    chars = []
    extended = False
    try:
        while True:
            await asyncio.sleep(0)
            if not msvcrt.kbhit():
                await asyncio.sleep(0.05)
                continue
            char = msvcrt.getwch()
            if extended:
                extended = False
                continue
            if char in ("\x00", "\xe0"):
                extended = True
                continue
            if char in ("\x03", "\x1a"):
                raise EOFError
            if char in ("\r", "\n"):
                return "".join(chars)
            if char == "\b":
                if chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
            elif char.isprintable() and len(chars) < 10000:
                chars.append(char)
                sys.stdout.write(char)
            sys.stdout.flush()
    except asyncio.CancelledError:
        while msvcrt.kbhit():
            msvcrt.getwch()
        raise
    finally:
        # No background reader survives cancellation or consumes the next command.
        sys.stdout.write("\n")
        sys.stdout.flush()
