from RlGlue import BaseEnvironment
from minatar import Environment

class Minatar(BaseEnvironment):
    def __init__(self, name, seed):
        self.env = Environment(name, random_seed=seed)

    def start(self):
        self.env.reset()
        s = self.env.state()
        extra = {
            'distance_rep': s
        }
        return s.astype('float32'), extra

    def step(self, a):
        r, t = self.env.act(a)
        sp = self.env.state().astype('float32')
        extra = {
            'distance_rep': sp
        }

        return (r, sp, t, extra)
