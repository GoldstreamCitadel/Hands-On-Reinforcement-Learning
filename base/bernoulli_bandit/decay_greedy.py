from solver import Solver
from bandit import BernoulliBandit
from greedy import plot_res
import numpy as np

class DecayGreedy(Solver):
    def __init__(self, bandit, init_prob=1.0):
        super(DecayGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob]*self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1/self.total_count: # linear decay epsilon
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += 1./(self.counts[k]+1) * (r-self.estimates[k])

        return k
    

def main():
    np.random.seed(42)
    bandit_10_arm = BernoulliBandit(10)
    decay_solver = DecayGreedy(bandit_10_arm)
    decay_solver.run(5000)
    print('dacay greedy cumulative regrets: ',decay_solver.regret)
    plot_res([decay_solver],["DecayGreedy"])


if __name__ == '__main__':
    main()