"""Screen capture and keyboard control for Overcooked! 2 (real game)."""
import time
import numpy as np
import mss
import cv2
from pynput.keyboard import Key, Controller

_KEYBOARD = Controller()

KEY_MAP = {
    "up": "w", "down": "s", "left": "a", "right": "d",
    "interact": "e", "chop_throw": "q",
}

# Discrete action set: each entry is a set of keys held simultaneously.
ACTIONS = [
    set(),                                  # 0 noop
    {"up"}, {"down"}, {"left"}, {"right"},  # 1-4 move
    {"interact"},                           # 5 interact
    {"chop_throw"},                         # 6 chop / throw
    {"up", "interact"}, {"down", "interact"},
    {"left", "interact"}, {"right", "interact"},   # 7-10 move+interact
    {"up", "chop_throw"}, {"down", "chop_throw"},
    {"left", "chop_throw"}, {"right", "chop_throw"}, # 11-14 move+chop
]
N_ACTIONS = len(ACTIONS)


class ScreenController:
    def __init__(self, monitor=1, capture_size=(1280, 720), resize_to=(320, 180),
                 keys=None, action_repeat_ms=100):
        self.monitor = monitor
        self.resize_to = tuple(resize_to)
        self.action_repeat = action_repeat_ms / 1000.0
        self.key_map = dict(KEY_MAP)
        if keys:
            self.key_map.update(keys)
        self._sct = mss.mss()
        mon = self._sct.monitors[monitor]
        w, h = capture_size
        # Center the capture box on the chosen monitor.
        left = mon["left"] + max(0, (mon["width"] - w) // 2)
        top = mon["top"] + max(0, (mon["height"] - h) // 2)
        self._region = {"left": left, "top": top, "width": w, "height": h}
        self._held = set()

    def grab(self):
        """Return (H, W, 3) uint8 BGR frame, resized."""
        shot = self._sct.grab(self._region)
        frame = np.asarray(shot)[:, :, :3]  # BGRA -> BGR
        return cv2.resize(frame, self.resize_to, interpolation=cv2.INTER_AREA)

    def _press(self, action_set):
        target = {self.key_map[k] for k in action_set}
        for k in self._held - target:
            _KEYBOARD.release(k)
        for k in target - self._held:
            _KEYBOARD.press(k)
        self._held = target

    def act(self, action_idx: int):
        self._press(ACTIONS[action_idx])
        time.sleep(self.action_repeat)

    def release_all(self):
        for k in list(self._held):
            _KEYBOARD.release(k)
        self._held = set()

    def close(self):
        self.release_all()
        self._sct.close()
