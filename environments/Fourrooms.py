from typing import Optional

import numpy as np

from glue.environment import BaseEnvironment
from PyRlEnvs.domains.GridWorld.Elements import WallState, StartState, GoalState
from PyRlEnvs.domains.GridWorld.GridWorld import GridWorldBuilder


WALL_COLOR = [0.8, 0.8, 0.8]
AGENT_COLOR = [1.0, 0.0, 0.0]
GOAL_COLOR = [0.0, 1.0, 0.0]


def build():
    builder = GridWorldBuilder((20, 20))
    builder.costToGoal = True
    builder.addElement(StartState((10, 10)))
    builder.addElement(GoalState((19, 19), -1))
   

    # # enclosed walls except two openning on top left
    builder.addElements([WallState((x,10)) for x in range(0,20) if x != 5 and x != 15])
    builder.addElements([WallState((10,y)) for y in range(0,20) if y != 5 and y != 15])

    return builder.build()

class Fourrooms(BaseEnvironment):
    def __init__(self, seed: int, max_steps: Optional[int] = None, distance_rep_size: int = 4):
        self.env = build()()

        self.seed = seed
        self.actions: int = 4

        #self.laplacian = self.compute_laplacian(rep_dim = distance_rep_size)
        #self._max_distance = self.find_max_distance()

    def rep(self, state: int):
        rep = np.zeros(self.env.num_states, dtype=np.float32)
        rep[state] = 1.0
        return rep

    def laplacian_rep(self,state, scale: bool):
        if scale:
            return self.laplacian[state]/self._max_distance

        return self.laplacian[state]

    def start(self):
        s = self.env.start()

        extra = {
            'distance_rep': s
        }
        return self.rep(s), extra

    def step(self, action: int):
        _, sp, t, extra = self.env.step(action)
        r = -1.0

        extra |= {
            'distance_rep': sp
        }
        return r, self.rep(sp), t, extra

    def compute_laplacian(self, rep_dim: int):
        random_policy = lambda _: np.ones(self.actions) / self.actions
        w = self.env.constructTransitionMatrix(random_policy)

        d = np.diag(np.sum(w, axis=1))
        l = d - w

        eigvals, eigvecs = np.linalg.eig(l)
        idx = np.argsort(eigvals)
        
        # print(eigvals)
        # print(eigvecs)
        # exit()

        #reorder eigvecs
        eigvecs = eigvecs[:,idx]
        eigvals = eigvals[idx]

        #first issue is that the eigenvalues of a laplacian matrix are always non-negative, since it
        #is positive definite. 

        #This means that there is something wrong with our w matrix.

        eigvecs = eigvecs[:,1:rep_dim+1]
        eigvals = eigvals[1:rep_dim+1]

        eigvecs = eigvecs/np.sqrt(eigvals).reshape(1,-1)
        return np.real(eigvecs)

    def find_max_distance(self):
        _max_distance = -float('inf')
        for x in range(self.env.num_states):
            for y in range(self.env.num_states):
                distance = np.linalg.norm(self.laplacian_rep(x, scale = False) - self.laplacian_rep(y, scale = False))
                if distance > _max_distance:
                    _max_distance = distance

        return _max_distance
