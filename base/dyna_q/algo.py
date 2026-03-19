import numpy as np
import random


class DynaQ:
    def __init__(self, ncol, nrow,
                 epsilon, alpha, gamma,
                 n_planning, n_action=4):
        self.Q_table = np.zeros([nrow*ncol, n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # planning: randomly review a past state, if n==0, it's q-learning.
        self.n_planning = n_planning # 1 q-learning, n q-planning
        self.model = dict() # env
        # dict once created (when agent created), always exists.


    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action
    

    def q_learning(self, s0, a0, r, s1):
        td_error = r + self.gamma * self.Q_table[s1].max(
        ) - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha*td_error


    def update(self, s0, a0, r, s1):
        self.q_learning(s0, a0, r, s1)
        self.model[(s0, a0)] = r, s1 # agent do a thing.
        for _ in range(self.n_planning):
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            self.q_learning(s, a, r, s_)