"""Run firmware.elf on a Renode board model and capture its console UART.

  python firmware/run_renode.py firmware/out/firmware.elf --board nano33ble [--timeout 300]

Prints the console and exits 0 when the firmware prints DONE, 1 on FAILED/timeout."""
import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BOARDS = {  # board -> (Renode platform description, console UART peripheral)
    "nano33ble": ("platforms/boards/arduino_nano_33_ble.repl", "sysbus.uart0"),
    "unor4": ("platforms/boards/arduino_uno_r4_minima.repl", "sysbus.sci2"),
}
RENODE_CANDIDATES = [r"C:\Program Files\Renode\renode.exe", "renode"]


def find_renode():
    for c in RENODE_CANDIDATES:
        if Path(c).exists() or shutil.which(c):
            return c
    raise FileNotFoundError("renode not found")


def resc(elf, log, board):
    repl, uart = BOARDS[board]
    p = lambda x: str(Path(x).resolve()).replace("\\", "/")  # noqa: E731
    return f"""
using sysbus
mach create "{board}"
machine LoadPlatformDescription @{repl}
sysbus LoadELF @{p(elf)}
{uart} CreateFileBackend @{p(log)} true
start
"""


def _read(path):
    try:
        return path.read_text(errors="replace") if path.exists() else ""
    except PermissionError:
        return ""


def run(elf, board, timeout=300):
    renode = find_renode()
    work = Path(tempfile.mkdtemp(prefix="renode_"))
    console, script = work / "sci2.log", work / "run.resc"
    script.write_text(resc(elf, console, board))
    proc = subprocess.Popen([renode, "--disable-gui", "--console", "--hide-log", "-e",
                             f"include @{str(script).replace(chr(92), '/')}"],
                            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0, text = time.time(), ""
    try:
        while time.time() - t0 < timeout:
            time.sleep(1)
            text = _read(console) or text
            if "DONE" in text or "FAILED" in text:
                break
            if proc.poll() is not None:
                break
    finally:
        proc.kill()
        proc.wait()
    text = _read(console) or text  # Windows: Renode may hold the file locked until it exits
    return text, "DONE" in text, time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("elf")
    ap.add_argument("--timeout", type=int, default=90,
                    help="Windows: Renode locks the log until it exits, so the run lasts this long")
    ap.add_argument("--out", help="also save the console log here")
    ap.add_argument("--board", choices=sorted(BOARDS), default="nano33ble")
    args = ap.parse_args(argv)
    text, ok, secs = run(args.elf, args.board, args.timeout)
    print(text.replace("\r\n", "\n"))
    print(f"[renode] {'DONE' if ok else 'NOT DONE'} after {secs:.0f}s wall time")
    if args.out:
        Path(args.out).write_text(text.replace("\r\n", "\n"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
