from PyRlEnvs.domains.MountainCar import GymMountainCar as Env

class MountainCar(Env):
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
