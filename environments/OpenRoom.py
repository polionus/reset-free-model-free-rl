import numpy as np
from typing import Optional
from collections.abc import Callable
import gymnasium as gym
from aim import Run

BUCKET_SIZE = 20


def identity_rep(s: np.ndarray) -> np.ndarray:
    return s


### So this function receives a (x, y) coordinate and turns it into a one-hot rep.
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


class OpenRoom(gym.Env):
    GOAL_BOX = np.array([[0.9, 1.0], [0.9, 1.0]])
    ACTION_DISPLACEMENT = 0.05

    def __init__(
        self,
        seed: int,
        mode: str,
        max_steps: Optional[int] = None,
        reset_free: bool = True,
        run: Optional[Run] =  None 
    ):
        self.seed = seed
        self.max_steps = max_steps
        self.actions: int = 4
        self.n_step = 0
        self.reset_free = reset_free
        self.run = run

        # self.observation_space = self.get_observations(mode)

        if mode == "continuous":
            self.observation_space = gym.spaces.Box(0, 1, shape=(2,))
        elif mode == "discrete":
            self.observation_space = gym.spaces.Box(0, 1, shape=(BUCKET_SIZE * 2,))
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self.action_space = gym.spaces.Discrete(4)

        self.s: np.ndarray = np.array([0.1, 0.1])
        self.rng = np.random.default_rng(seed)

        self.rep = self.get_rep(mode)

    def in_goal(self, s: np.ndarray) -> bool:
        return bool(
            np.all(s >= OpenRoom.GOAL_BOX[:, 0])
            and np.all(s <= OpenRoom.GOAL_BOX[:, 1])
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.n_step = 1
        self.s = np.array([0.1, 0.1])
        info = {}
        return self.rep(self.s), info

    def step(self, action):
        self.n_step += 1

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
        sp = sp + self.rng.uniform(0.0, 0.01, size=2)
        sp = np.clip(sp, 0.0, 1.0)

        

        reward = 0.0
        t = False
        if self.in_goal(sp) and not self.reset_free:
            reward = 1.0
            t = True

        self.s = sp

        trunc = (
            False
            if self.max_steps is None or self.reset_free
            else self.n_step >= self.max_steps
        )

        info = {}
        if t or trunc:
            info["final_observation"] = self.rep(self.s)

        if self.run is not None: 
            self.run.track(reward, name = "Reward")
            
            if t or trunc: 
                self.run.track(reward, name="Return")

        return self.rep(self.s), reward, t, trunc, info

    def get_rep(self, mode: str) -> Callable:
        if mode == "continuous":
            return identity_rep
        elif mode == "discrete":
            return desc_rep
        else:
            raise ValueError(f"Invalid mode: {mode}")
