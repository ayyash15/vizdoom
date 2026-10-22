"""Thin ViZDoom wrapper: grayscale, resize, frame-skip, frame-stack, discrete actions.

combo=True adds one extra action per non-last button that presses that button AND the last
button (ATTACK in basic) together: basic -> LEFT, RIGHT, ATTACK, LEFT+ATTACK, RIGHT+ATTACK.
"""
import os
from collections import deque

import cv2
import numpy as np
import vizdoom as vzd


class DoomEnv:
    def __init__(self, scenario="basic", frame_skip=4, size=64, stack=4,
                 render=False, seed=None, combo=False):
        self.size, self.stack, self.frame_skip = size, stack, frame_skip
        game = vzd.DoomGame()
        cfg = os.path.join(vzd.scenarios_path, f"{scenario}.cfg")
        if not os.path.exists(cfg):
            raise FileNotFoundError(f"Scenario config not found: {cfg}")
        game.load_config(cfg)
        game.set_window_visible(render)
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        game.set_episode_timeout(game.get_episode_timeout() or 2100)
        if seed is not None:
            game.set_seed(seed)
        game.init()
        self.game = game
        n = game.get_available_buttons_size()
        acts = [row for row in np.eye(n, dtype=np.int32)]          # one button per action
        if combo and n >= 2:
            for i in range(n - 1):                                  # button i + last button
                v = np.zeros(n, dtype=np.int32)
                v[i] = 1
                v[n - 1] = 1
                acts.append(v)
        self.actions = [a.tolist() for a in acts]
        self.n_actions = len(self.actions)
        self.frames = deque(maxlen=stack)
        self._last_raw = np.zeros((120, 160), dtype=np.uint8)

    def _prep(self, raw):
        return cv2.resize(raw, (self.size, self.size), interpolation=cv2.INTER_AREA)

    def _obs(self):
        return np.stack(self.frames, axis=0)          # (stack, H, W) uint8

    def reset(self):
        self.game.new_episode()
        raw = self.game.get_state().screen_buffer
        self._last_raw = raw
        f = self._prep(raw)
        for _ in range(self.stack):
            self.frames.append(f)
        return self._obs()

    def step(self, action):
        reward = self.game.make_action(self.actions[action], self.frame_skip)
        done = self.game.is_episode_finished()
        if not done:
            self._last_raw = self.game.get_state().screen_buffer
        self.frames.append(self._prep(self._last_raw))
        return self._obs(), float(reward), done

    def raw_frame(self):
        return self._last_raw

    def close(self):
        self.game.close()
