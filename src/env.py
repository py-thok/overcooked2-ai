"""Gymnasium environment wrapping the real Overcooked! 2 game.

Observation: stacked grayscale frames (frame_stack, H, W).
Reward: change in score region pixels -> OCR-free proxy, or shaped
        heuristic if you enable detection.  Simplest robust default:
        sparse reward from score HUD via template change detection.

You must position the game window so the score HUD is visible inside
the capture region, and set `score_roi` to the pixel box of the score
digits (x, y, w, h) in the *resized* frame.
"""
import time
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2

from screen_io import ScreenController, N_ACTIONS


class OvercookedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, screen_cfg, control_cfg, agent_cfg,
                 score_roi=None, episode_time_s=None):
        super().__init__()
        self.ctrl = ScreenController(
            monitor=screen_cfg["monitor"],
            capture_size=tuple(screen_cfg["capture_size"]),
            resize_to=tuple(screen_cfg["resize_to"]),
            keys=control_cfg.get("keys"),
            action_repeat_ms=control_cfg["action_repeat_ms"],
        )
        h, w = screen_cfg["resize_to"][1], screen_cfg["resize_to"][0]
        self.frame_stack = agent_cfg["frame_stack"]
        self.step_delay = agent_cfg["step_delay_s"]
        self.max_steps = agent_cfg["max_episode_steps"]
        self.score_roi = score_roi  # (x, y, w, h) in resized frame or None
        self.episode_time_s = episode_time_s

        self.observation_space = spaces.Box(
            0, 255, shape=(self.frame_stack, h, w), dtype=np.uint8)
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._frames = deque(maxlen=self.frame_stack)
        self._steps = 0
        self._prev_score_sig = None
        self._t0 = None

    # ------------------------------------------------------------------
    def _gray(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _score_signature(self, gray):
        """Hash of the score HUD pixels; changes when the score changes."""
        if self.score_roi is None:
            return None
        x, y, w, h = self.score_roi
        crop = gray[y:y + h, x:x + w]
        _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bw.tobytes()

    def _obs(self, frame):
        g = self._gray(frame)
        self._frames.append(g)
        return np.stack(list(self._frames), axis=0)

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ctrl.release_all()
        self._frames.clear()
        self._steps = 0
        self._t0 = time.time()
        frame = self.ctrl.grab()
        obs = None
        for _ in range(self.frame_stack):
            obs = self._obs(frame)
        self._prev_score_sig = self._score_signature(self._frames[-1])
        return obs, {}

    def step(self, action):
        self.ctrl.act(int(action))
        if self.step_delay:
            time.sleep(self.step_delay)
        frame = self.ctrl.grab()
        obs = self._obs(frame)
        self._steps += 1

        reward = 0.0
        sig = self._score_signature(self._frames[-1])
        if sig is not None and self._prev_score_sig is not None and sig != self._prev_score_sig:
            reward = 1.0  # score changed (up or down); refine with OCR later
        self._prev_score_sig = sig

        terminated = False
        truncated = self._steps >= self.max_steps
        if self.episode_time_s and (time.time() - self._t0) > self.episode_time_s:
            truncated = True
        return obs, reward, terminated, truncated, {}

    def close(self):
        self.ctrl.close()
