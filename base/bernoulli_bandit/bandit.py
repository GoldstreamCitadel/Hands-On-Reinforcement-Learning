import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    def __init__(self, K):
        self.probs = np.random.uniform(size=K)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.K = K

    def step(self, K):
        if np.random.rand() < self.probs[K]:
            return 1
        else:
            return 0
        
def main():
    np.random.seed(1)
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    print("Randomly generate a %d-arm bandit."%K)
    print("Max-prob-idx:%d, prob-award:%.4f"%\
        (bandit_10_arm.best_idx, bandit_10_arm.best_prob))

if __name__ == '__main__':
    main()