"""Game-instance watchdog: reconnect the bridge, relaunch the game if dead.

Two crashes in a day (run69 ConnectionReset, run70 ConnectionAborted) killed
the trainer with hours of downtime each. The games die on their own
schedule; the trainer must survive it: detect the dead bridge, relaunch the
matching instance (port 8765 = original install, 8767 = the copy), wait for
the bridge, reconnect.

Launch goes through powershell Start-Process: direct exec
(ProcessStartInfo/Popen) makes the game exit 53 instantly (Steam shim).
"""
import os
import socket
import subprocess
import time

GAME_DIRS = {
    8765: (r"E:\SteamLibrary\steamapps\common\Overcooked! 2", None),
    8767: (r"E:\SteamLibrary\steamapps\common\Overcooked2_inst2", "8767"),
}


def bridge_alive(port, timeout=2.0):
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout)
        s.close()
        return True
    except OSError:
        return False


def ensure_game(port, wait_s=300):
    """Return when the bridge answers on port, relaunching the game first
    if it's down. Raises ConnectionError if it never comes back."""
    if bridge_alive(port):
        return
    d, penv = GAME_DIRS[port]
    print(f"[watchdog] game for port {port} dead — relaunching {d}")
    cmd = f"Start-Process '{os.path.join(d, 'Overcooked2.exe')}' -WorkingDirectory '{d}'"
    env = dict(os.environ)
    if penv:
        cmd = f"$env:OC2BRIDGE_PORT='{penv}'; {cmd}; Remove-Item Env:OC2BRIDGE_PORT"
    subprocess.Popen(["powershell", "-Command", cmd], env=env)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if bridge_alive(port):
            print(f"[watchdog] port {port} back after {time.time()-t0:.0f}s")
            return
        time.sleep(2)
    raise ConnectionError(f"game on port {port} did not come back in {wait_s}s")
