import numpy as np
from typing import Optional
from collections.abc import Callable
from glue.environment import BaseEnvironment

BUCKET_SIZE = 20

def identity_rep(s: np.ndarray) -> np.ndarray:
    return s

def desc_rep(s: np.ndarray) -> np.ndarray:
    x_bucket = int(s[0] * BUCKET_SIZE)
    y_bucket = int(s[1] * BUCKET_SIZE)

    if x_bucket == BUCKET_SIZE:
        x_bucket -= 1

    if y_bucket == BUCKET_SIZE:
        y_bucket -= 1


    # Convert to a one-hot representation
    rep = np.zeros(BUCKET_SIZE * 2)
    rep[x_bucket] = 1
    rep[y_bucket + BUCKET_SIZE] = 1

    return rep


class LargeRoom(BaseEnvironment):
    GOAL_BOX = np.array([[0.9, 1.0], [0.9, 1.0]])
    ACTION_DISPLACEMENT = 0.01

    def __init__(self, seed: int, mode: str, max_steps: Optional[int] = None):
            self.seed = seed
            self.max_steps = max_steps
            self.actions: int = 4

            self.observations = self.get_observations(mode)
            self.rep = self.get_rep(mode)

            self.s: np.ndarray = np.array([0.1, 0.1])
            self.rng = np.random.default_rng(seed)

    def in_goal(self, s: np.ndarray) -> bool:
        return bool(
            np.all(s >= LargeRoom.GOAL_BOX[:, 0]) and
            np.all(s <= LargeRoom.GOAL_BOX[:, 1])
        )

    def start(self):
            self.s = np.array([0.1, 0.1])
            extra = {
                'distance_rep': self.s
            }
            return self.rep(self.s), extra

    def step(self, action):

        # Calculate the next state
        sp = self.s.copy()
        if action == 0:
            sp[0] -= self.ACTION_DISPLACEMENT
        elif action == 1:
            sp[0] += self.ACTION_DISPLACEMENT
        elif action == 2:
            sp[1] -= self.ACTION_DISPLACEMENT
        elif action == 3:
            sp[1] += self.ACTION_DISPLACEMENT
        sp = sp + self.rng.uniform(0.0, self.ACTION_DISPLACEMENT * 0.2, size=2)
        sp = np.clip(sp, 0.0, 1.0)

        reward = 0.0
        t = False
        if self.in_goal(sp):
            reward = 1.0
            t = True

        self.s = sp

        extra = {
            'distance_rep': self.s
        }

        return reward, self.rep(sp), t, extra

    def get_observations(self, mode: str) -> tuple[int, ...]:
        if mode == 'continuous':
            return (2,)
        elif mode == 'discrete':
            return (2 * BUCKET_SIZE,)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def get_rep(self, mode: str) -> Callable:
        if mode == 'continuous':
            return identity_rep
        elif mode == 'discrete':
            return desc_rep
        else:
            raise ValueError(f"Invalid mode: {mode}")
