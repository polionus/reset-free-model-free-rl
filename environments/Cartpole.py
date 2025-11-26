from PyRlEnvs.domains.Cartpole import Cartpole as Env

class Cartpole(Env):
    def start(self):
        s = super().start()

        extra = {
            'distance_rep': s
        }
        return s, extra

    def step(self, a):
        r, sp, t, extra = super().step(a)

        extra |= {
            'distance_rep': sp
        }
        return (r, sp, t, extra)
