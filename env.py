"""Thin ViZDoom wrapper: grayscale, resize, frame-skip, frame-stack, discrete actions.

aim_mode="strafe" (default): original behaviour. One action per button of the scenario cfg
  (basic: MOVE_LEFT, MOVE_RIGHT, ATTACK). combo=True adds LEFT+ATTACK, RIGHT+ATTACK.
aim_mode="turn":  buttons overridden to TURN_LEFT, TURN_RIGHT, ATTACK (reward/termination untouched).
aim_mode="delta": buttons overridden to TURN_LEFT_RIGHT_DELTA, ATTACK. Actions = signed deltas
  (+/- aim_deltas * aim_unit), ATTACK, and (combo=True) each signed delta together with ATTACK.
  The delta is applied on the FIRST tic of a decision only; remaining frame_skip-1 tics repeat just the
  ATTACK component, so frame_skip does not multiply the turn angle.
ATTACK is always the LAST button, so the combo logic is the same in every mode.
"""
import os
from collections import deque

import cv2
import numpy as np
import vizdoom as vzd


class DoomEnv:
    def __init__(self, scenario="basic", frame_skip=4, size=64, stack=4,
                 render=False, seed=None, combo=False,
                 aim_mode="strafe", aim_deltas="1,3,8,20", aim_unit=1.0):
        self.size, self.stack, self.frame_skip = size, stack, frame_skip
        self.aim_mode = aim_mode
        game = vzd.DoomGame()
        cfg = os.path.join(vzd.scenarios_path, f"{scenario}.cfg")
        if not os.path.exists(cfg):
            raise FileNotFoundError(f"Scenario config not found: {cfg}")
        game.load_config(cfg)
        if aim_mode == "turn":
            game.clear_available_buttons()
            for b in (vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK):
                game.add_available_button(b)
        elif aim_mode == "delta":
            game.clear_available_buttons()
            game.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA)
            game.add_available_button(vzd.Button.ATTACK)
            # UNVERIFIED: default clipping of delta buttons; raise it so the largest delta is not clipped.
            game.set_button_max_value(vzd.Button.TURN_LEFT_RIGHT_DELTA, 180)
        elif aim_mode != "strafe":
            raise ValueError(f"unknown aim_mode {aim_mode}")
        game.set_window_visible(render)
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        game.set_episode_timeout(game.get_episode_timeout() or 2100)
        if seed is not None:
            game.set_seed(seed)
        game.init()
        self.game = game
        n = game.get_available_buttons_size()
        self.n_buttons = n
        if aim_mode == "delta":
            ds = [float(x) for x in str(aim_deltas).split(",") if x.strip()]
            signed = [-d for d in reversed(ds)] + ds
            acts = []
            for d in signed:                                        # turn only
                v = [0.0] * n; v[0] = d * aim_unit; acts.append(v)
            v = [0.0] * n; v[1] = 1.0; acts.append(v)               # ATTACK only
            if combo:
                for d in signed:                                    # turn + ATTACK in the same tic
                    v = [0.0] * n; v[0] = d * aim_unit; v[1] = 1.0; acts.append(v)
            self.actions = acts
        else:
            acts = [row for row in np.eye(n, dtype=np.int32)]       # one button per action
            if combo and n >= 2:
                for i in range(n - 1):                              # button i + last button
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
        act = self.actions[action]
        if self.aim_mode == "delta" and self.frame_skip > 1:
            reward = self.game.make_action(act, 1)                  # delta applied once
            if not self.game.is_episode_finished():
                hold = [0.0] * self.n_buttons
                hold[-1] = act[-1]                                  # keep ATTACK held, no more turning
                reward += self.game.make_action(hold, self.frame_skip - 1)
        else:
            reward = self.game.make_action(act, self.frame_skip)
        done = self.game.is_episode_finished()
        if not done:
            self._last_raw = self.game.get_state().screen_buffer
        self.frames.append(self._prep(self._last_raw))
        return self._obs(), float(reward), done

    def raw_frame(self):
        return self._last_raw

    def close(self):
        self.game.close()
