from typing import Optional
import gymnasium
from glue.environment import BaseEnvironment

class Gym(BaseEnvironment):
    def __init__(self, name: str, seed: int, max_steps: Optional[int] = None):
        self.env = gymnasium.make(name, max_episode_steps=max_steps)
        self.seed = seed

        self.max_steps = max_steps

    def start(self):
        self.seed += 1
        s, extra = self.env.reset(seed=self.seed)

        extra |= {
            'distance_rep': s
        }
        return s, extra

    def step(self, a):
        sp, r, t, _, extra = self.env.step(a)

        extra |= {
            'distance_rep': sp
        }
        return (r, sp, t, extra)
