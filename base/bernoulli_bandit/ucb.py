import numpy as np
from solver import Solver
from bandit import BernoulliBandit
from greedy import plot_res

class UCB(Solver):
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob]*self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count)/(2*(self.counts+1))
        )
        k = np.argmax(ucb)
        r = self.bandit.step(k)
        self.estimates[k] += 1./(self.counts[k]+1)*(r-self.estimates[k])
        return k
    

def main():
    bandit_10_arm = BernoulliBandit(10)
    np.random.seed(42)
    coef = 1
    UCB_solver = UCB(bandit_10_arm, coef)
    UCB_solver.run(5000)
    print('Upper Confidence Bound cumulative regret: ',UCB_solver.regret)
    plot_res([UCB_solver],["UCB"])

if __name__ == '__main__':
    main()
