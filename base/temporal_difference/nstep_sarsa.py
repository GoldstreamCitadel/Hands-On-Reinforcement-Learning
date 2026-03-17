import numpy as np
from tqdm import tqdm
from sarsa import CliffEnv, print_agent
import matplotlib.pyplot as plt

class nstep_Sarsa:
    def __init__(self, n, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow*ncol, n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n # n-step Sarsa
        self.state_list = []
        self.action_list = []
        self.reward_list = []

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action
    
    def best_action(self, state): # print strategy
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a
    
    def update(self, s0, a0, r, s1, a1, done):
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)
        if len(self.state_list) == self.n:
            G = self.Q_table[s1, a1]
            for i in reversed(range(self.n)):
                G = self.gamma*G + self.reward_list[i]
                if done and i>0:
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.Q_table[s, a] += self.alpha *(G-self.Q_table[s,a])
            s = self.state_list.pop(0)
            a = self.action_list.pop(0)
            self.reward_list.pop(0)

            self.Q_table[s, a] += self.alpha*(G-self.Q_table[s, a])
        if done:
            self.state_list=[]
            self.action_list=[]
            self.reward_list=[]


class CliffEnvTrainer:
    def __init__(self, n_step, ncol, nrow, alpha, epsilon, gamma, num_episodes):
        self.n_step = n_step
        self.ncol = ncol
        self.nrow = nrow
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_episodes = num_episodes

    def train(self):
        
        """
        n_step = n_step
        alpha = alpha
        epsilon = epsilon
        gamma = gamma 
        ncol = ncol
        nrow = nrow
        num_episodes = num_episodes
        """
        # upper graph DO NOT SAVE
        # lower-layer cover upper-layer, var init fail!!!
        num_episodes = self.num_episodes
        n_step = self.n_step
        env = CliffEnv(self.ncol, self.nrow)
        agent = nstep_Sarsa(self.n_step, self.ncol, self.nrow, self.epsilon, self.alpha, self.gamma)
        

        return_list = []
        for i in range(10):
            with tqdm(total=int(num_episodes/10), desc='Iteration %d'%i) as pbar:
                for i_episode in range(int(num_episodes/10)):
                    episode_return = 0
                    state = env.reset()
                    action = agent.take_action(state)
                    done = False
                    while not done:
                        next_state, reward, done = env.step(action)
                        next_action = agent.take_action(next_state)
                        episode_return += reward # tmply not decay
                        agent.update(
                            state, action, reward,
                            next_state, next_action, done
                        )
                        state, action = next_state, next_action
                    return_list.append(episode_return)
                    if (i_episode + 1)%10 == 0:
                        pbar.set_postfix({
                            'episode':
                            '%d'%(num_episodes/10*i+i_episode+1),
                            'return':
                            '%.3f'%np.mean(return_list[-10:])
                        })
                    pbar.update(1)

        episodes_list = list(range(len(return_list)))
        plt.plot(episodes_list, return_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('{}-step Sarsa on {}'.format(n_step, 'CliffWalking'))
        plt.show()

        action_meaning = ['^','v','<','>']
        print("Sarsa-algorithm, final strategy:")
        print_agent(agent, env, action_meaning, list(range(37, 47)),[47])


if __name__ == '__main__':
    library_crush = CliffEnvTrainer(
        n_step=5
        , ncol=12
        , nrow=4
        , alpha=0.1
        , epsilon=0.1
        , gamma=0.9
        , num_episodes=500
    )
    np.random.seed(0)
    library_crush.train()

