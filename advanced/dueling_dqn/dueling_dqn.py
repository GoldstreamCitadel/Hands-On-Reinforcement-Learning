import torch
import torch.nn.functional as F
import sys
import numpy as np

sys.path.append("..")
from DQN import rl_utils
from DQN_plus.double_dqn import Qnet

class VAnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim) # shared
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A - A.mean(1).view(-1, 1)
        return Q
    

class DQN:
    # including double-dqn & dueling-dqn
    # 和单dqn仅网络架构不同，训练方法仍是vanilla-dqn
    def __init__(self,
                 state_dim, hidden_dim, action_dim,
                 learning_rate, gamma, epsilon,
                 target_update, device, dqn_type='VanillaDQN'):
        self.action_dim = action_dim

        if dqn_type == 'DuelingDQN':
            self.q_net = VAnet(state_dim, hidden_dim,
                               self.action_dim).to(device)
            self.target_q_net = VAnet(state_dim, hidden_dim,
                                      self.action_dim).to(device)
        else:
            self.q_net = Qnet(state_dim, hidden_dim,
                              self.action_dim).to(device)
            self.target_q_net = Qnet(state_dim, hidden_dim,
                              self.action_dim).to(device)
            
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.dqn_type = dqn_type
        self.device = device

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            return self.q_net(state).max().item()
        
    # 这段儿得加，否则agent类型就没有max_q_value这个func，想用隔壁双dqn的train函数就用不了
    def max_q_value(self, state):
        state = np.array(state)
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()
    
        
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict[actions]).view(-1, 1).to(
            self.device
        )
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        
        q_values = self.q_net(states).gather(1, actions)
        # 原生dqn网络 Q值计算依靠 目标网络
        # 双dqn网络 Q值计算依靠 原来那个干净的网络
        if self.dqn_type == 'DoubleDQN':
            max_action = self.q_net(next_states).max(1)[1].view(
                -1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(
                1, max_action)
        else:
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
                -1, 1
            )

        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count +=1


# 得一并迁过来，否则nonetype报错，离散动作对不上
def dis_to_con(discrete_action, env, action_dim): # discrete move back to continuous func
    action_lowbound = env.action_space.low[0]
    action_upbound = env.action_space.high[0]
    return action_lowbound + (discrete_action /
                            (action_dim - 1)) * (action_upbound -
                                                action_lowbound)
    
# 就不导了，直接迁移，看看对不对
def train_DQN(agent, env, num_episodes,
              replay_buffer, minimal_size, batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d'% i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(
                        state) * 0.005 + max_q_value * 0.995 # 平滑
                    max_q_value_list.append(max_q_value)

                    action_continuous = dis_to_con(action, env, agent.action_dim)

                    next_state, reward, done, _ = env.step([action_continuous])
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(
                            batch_size
                        )
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if(i_episode + 1)%10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d'%(num_episodes/10*i+i_episode+1)
                    })
                pbar.update(1)
    return return_list, max_q_value_list



if __name__ == '__main__':
    import random
    from DQN import rl_utils
    from tqdm import tqdm
    import gym

    lr = 1e-2
    num_episodes = 200
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.1
    target_update = 50
    buffer_size = 5000
    minimal_size = 1000
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")
    
    env_name = 'Pendulum-v0' # 小三蹦子甩棍儿
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 11 # 离散动作,还可以拆
 

    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device, 'DuelingDQN') # 默认Vanilla 传DuelingDQN
    
    return_list, max_q_value_list = train_DQN(agent, env, num_episodes,
                                          replay_buffer, minimal_size,
                                          batch_size)


    import os
    import matplotlib.pyplot as plt
    graph_dir = './graph'
    os.makedirs(graph_dir, exist_ok=True)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']

    episodes_list = list(range(len(return_list)))
    mv_return = rl_utils.moving_average(return_list, 5)

    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Dueling_DQN on {}\n{}'.format(env_name, "学不同动作差异性，最大回报更高"))
    fig1_path = os.path.join(graph_dir, f'dueling_dqn_{env_name}_ret.png')
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"已保存: {fig1_path}")
    plt.show()

    frames_list = list(range(len(max_q_value_list)))
    plt.plot(frames_list, max_q_value_list)
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title('Dueling_DQN on {}\n{}'.format(env_name, "训练更稳定"))
    fig2_path = os.path.join(graph_dir, f'dueling_dqn_{env_name}_Q.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"已保存: {fig2_path}")
    plt.show()

    