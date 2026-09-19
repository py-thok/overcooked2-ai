"""Bring the Overcooked! 2 window to the foreground and send key presses.

Used to get past the StartScreen engagement prompt ("press any key")
without human input, so level loading can be fully autonomous.
"""
import ctypes
import time
from ctypes import wintypes

user32 = ctypes.windll.user32

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
VK_SPACE = 0x20
VK_RETURN = 0x0D


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", INPUT_UNION)]


def find_game_window(title_part="Overcooked"):
    """Return hwnd of the game window, or None."""
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def enum_cb(hwnd, lparam):
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                if title_part.lower() in buf.value.lower():
                    found.append(hwnd)
        return True

    user32.EnumWindows(enum_cb, 0)
    return found[0] if found else None


def force_foreground(hwnd):
    """SetForegroundWindow with the AttachThreadInput workaround."""
    fg = user32.GetForegroundWindow()
    cur_tid = ctypes.windll.kernel32.GetCurrentThreadId()
    fg_tid = user32.GetWindowThreadProcessId(fg, None)
    target_tid = user32.GetWindowThreadProcessId(hwnd, None)
    user32.AttachThreadInput(cur_tid, fg_tid, True)
    user32.AttachThreadInput(cur_tid, target_tid, True)
    try:
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        user32.AttachThreadInput(cur_tid, fg_tid, False)
        user32.AttachThreadInput(cur_tid, target_tid, False)


def send_key(vk=VK_SPACE, hold_s=0.05):
    """Inject a key press at system level (goes to the focused window)."""
    down = INPUT(type=INPUT_KEYBOARD,
                 u=INPUT_UNION(ki=KEYBDINPUT(vk, 0, 0, 0, None)))
    up = INPUT(type=INPUT_KEYBOARD,
               u=INPUT_UNION(ki=KEYBDINPUT(vk, 0, KEYEVENTF_KEYUP, 0, None)))
    arr = (INPUT * 2)(down, up)
    user32.SendInput(2, arr, ctypes.sizeof(INPUT))
    time.sleep(hold_s)


def press_in_game(vk=VK_SPACE, settle_s=0.5):
    """Focus the game window and press a key. Returns True if window found."""
    hwnd = find_game_window()
    if not hwnd:
        return False
    force_foreground(hwnd)
    time.sleep(settle_s)
    send_key(vk)
    return True


if __name__ == "__main__":
    print("game window:", find_game_window())
    print("press:", press_in_game(VK_SPACE))
