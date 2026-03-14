from solver import Solver
from bandit import BernoulliBandit
import numpy as np
from greedy import plot_res

class ThompsonSampling(Solver):
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K) # bonus-1-counts
        self._b = np.ones(self.bandit.K) # bonus-0-counts

    def run_one_step(self):
        # new step, new sample
        # ensure the newest _a , _b
        samples = np.random.beta(self._a, self._b) # sampling
        k = np.argmax(samples)
        r = self.bandit.step(k)

        self._a[k] += r
        self._b[k] += (1-r)
        return k
    

def main():
    bandit_10_arm = BernoulliBandit(10)
    np.random.seed(32)
    thompson_solver = ThompsonSampling(bandit_10_arm)
    thompson_solver.run(5000)
    print('thompson cumulative regret: ',thompson_solver.regret)
    plot_res([thompson_solver],['thompson'])


if __name__ == '__main__':
    main()