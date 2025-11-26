from typing import Optional

import numpy as np

from glue.environment import BaseEnvironment
from PyRlEnvs.domains.GridWorld.Elements import StartState, GoalState
from PyRlEnvs.domains.GridWorld.GridWorld import GridWorldBuilder

def build():
    builder = GridWorldBuilder((20, 20))
    builder.costToGoal = False
    builder.addElement(StartState((0, 0)))
    builder.addElement(GoalState((19, 19), 1))

    return builder.build()

class TabularOpenRoom(BaseEnvironment):
    def __init__(self, seed: int, max_steps: Optional[int] = None):
        self.env = build()()
        self.max_steps = max_steps

    def rep(self, state: int):
        rep = np.zeros(self.env.num_states, dtype=np.float32)
        rep[state] = 1.0
        return rep

    def get_eucledian_rep(self, state: int):
        x, y = self.env.getCoords(state)
        return np.array([x / self.env.shape[0], y / self.env.shape[1]], dtype=np.float32)

    def start(self):
        x = self.env.start()

        extra = {
            'distance_rep': self.get_eucledian_rep(x)
        }
        return self.rep(x), extra

    def step(self, action: int):
        r, sp, t, extra = self.env.step(action)

        extra |= {
            'distance_rep': self.get_eucledian_rep(sp)
        }
        return r, self.rep(sp), t, extra
