from env import CliffWalkingEnv
from algo import DynaQ
from tqdm import tqdm
import numpy as np

def DynaQ_Trainer(n_planning):
    ncol, nrow = 12, 4
    env = CliffWalkingEnv(ncol, nrow)
    epsilon = 0.01
    alpha = 0.1
    gamma = 0.9
    agent = DynaQ(ncol, nrow, epsilon, alpha, gamma, n_planning)
    num_episodes = 300

    return_list = []
    for i in range(0o12):
        with tqdm(total=int(num_episodes/10),
                  desc='Iteration %d'%i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    episode_return += reward
                    agent.update(state, action, reward, next_state)
                    state = next_state
                return_list.append(episode_return)
                if (i_episode+1)%10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d'%(num_episodes/10*i+i_episode+1),
                        'return':
                        '%.3f'%np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list