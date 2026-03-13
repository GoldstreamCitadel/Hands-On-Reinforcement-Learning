from solver import Solver
from bandit import BernoulliBandit
import numpy as np
import matplotlib.pyplot as plt

class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1./(self.counts[k]+1)*(r-self.estimates[k])
        return k
    

def plot_res(solvers, solver_names):
    assert len(solvers)==len(solver_names)
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, \
                 label=solver_names[idx])
    plt.xlabel('Time Steps')
    plt.ylabel("Cumulative regrets")
    # only change epsilon. arms equal.
    plt.title('%d-armed bandit'%solvers[0].bandit.K)
    plt.legend()
    plt.show()

def main():
    while(1):
        mode = str(input("$ Please choose your training mode. \
                    \n>>> input 1 for default epsilon value -- 0.01.\
                    \n>>> input 2 for multi epsilon values.\
                    \n>>> now input:"))
        bandit_10_arm = BernoulliBandit(10)

        if mode=='1':
            np.random.seed(1)
            epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
            epsilon_greedy_solver.run(5000)
            print('epsilon-greedy cumulative regret:')
            print(epsilon_greedy_solver.regret)
            plot_res([epsilon_greedy_solver], ["EpsilonGreedy"])
            break

        elif mode=='2':
            np.random.seed(0)
            epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
            solver_list = [
                EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons
            ]
            solver_names = ["epsilon={}, group:{}".format(e,i+1) for i,e in enumerate(epsilons)]
            for solver in solver_list:
                solver.run(5000)
            plot_res(solver_list, solver_names)
            break
        
        else:
            print("Invalid input. Try again.")
            continue

if __name__ == '__main__':
    main()