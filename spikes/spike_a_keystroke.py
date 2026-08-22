"""Spike A - prove macOS can inject keystrokes into a third-party app.

Throwaway. Answers one question: is Accessibility permission actually granted
to the app hosting this process? Without it pyautogui fails *silently* - no
error, no keystroke - which is the single most common "it doesn't work" in
this project.

    python spikes/spike_a_keystroke.py            # trust check only (safe)
    python spikes/spike_a_keystroke.py --press    # + countdown, then press Right

Permission is granted to the HOST APP (Terminal/iTerm/Cursor), never to Python
itself, and the host must be fully quit and reopened afterwards.
"""

import os
import subprocess
import sys
import time

# Default 0.1s sleep after EVERY call would eat half the project's 200ms budget.
import pyautogui

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False


def process_ancestry(pid=None, limit=6):
    """Walk up the parent-PID chain. The app to grant Accessibility to is in here."""
    pid = pid or os.getpid()
    chain = []
    for _ in range(limit):
        try:
            out = subprocess.run(
                ["ps", "-o", "ppid=,comm=", "-p", str(pid)],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            if not out:
                break
            ppid, _, comm = out.partition(" ")
            chain.append((pid, comm.strip()))
            pid = int(ppid)
            if pid <= 1:
                break
        except (subprocess.SubprocessError, ValueError):
            break
    return chain


def is_trusted():
    """AXIsProcessTrusted() - the authoritative answer, no user interaction needed."""
    try:
        from ApplicationServices import AXIsProcessTrusted
    except ImportError:
        return None
    return bool(AXIsProcessTrusted())


def main():
    print("Spike A - keystroke injection\n")

    print("Process ancestry (grant Accessibility to the app named here):")
    for pid, comm in process_ancestry():
        print(f"    {pid:>7}  {comm}")

    trusted = is_trusted()
    print()
    if trusted is None:
        print("  ? AXIsProcessTrusted() unavailable (pyobjc missing)")
    elif trusted:
        print("  PASS  AXIsProcessTrusted() = True - Accessibility is granted.")
    else:
        print("  FAIL  AXIsProcessTrusted() = False - keystrokes will NOT land.")
        print("        System Settings > Privacy & Security > Accessibility,")
        print("        enable the host app above, then FULLY QUIT and reopen it.")

    if "--press" not in sys.argv:
        print("\nTrust check only. Re-run with --press for the end-to-end test.")
        return 0 if trusted else 1

    if not trusted:
        print("\nRefusing --press: not trusted, the result would be meaningless.")
        return 1

    print("\nFocus your presentation NOW (Keynote / Slides / a PDF).")
    for n in range(5, 0, -1):
        print(f"  pressing Right in {n}...", flush=True)
        time.sleep(1)

    t0 = time.perf_counter()
    pyautogui.press("right")
    dispatch_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  sent 'right' in {dispatch_ms:.1f}ms")
    print("  PASS if the slide advanced. If not, the grant is on the wrong app.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
