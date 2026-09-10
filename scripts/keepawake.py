"""Run a command while holding a Windows 'system required' power request, so an
idle timer cannot put the machine to sleep mid-run. Does not (cannot) stop a
user-initiated sleep or a lid close.

    .venv\\Scripts\\python.exe scripts\\keepawake.py bash scripts/run_infovqa_colsmol.sh
"""
from __future__ import annotations

import ctypes
import subprocess
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    k32 = ctypes.windll.kernel32 if sys.platform == "win32" else None
    if k32:
        k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
    try:
        return subprocess.call(sys.argv[1:])
    finally:
        if k32:
            k32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    raise SystemExit(main())
