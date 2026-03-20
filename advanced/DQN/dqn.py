import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils

"""
# .gather方法使用指导
# 原始数据
t = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])  # 形状: [3, 3]

# 示例1.1: 在列方向收集 (dim=1)
index1 = torch.tensor([
    [0, 1],  # 第0行: 取第0列和第1列
    [1, 2],  # 第1行: 取第1列和第2列
    [2, 0]   # 第2行: 取第2列和第0列
])

result1 = torch.gather(t, 1, index1)
# 结果:
# [[1, 2],  # 第0行: 原始t[0,0]=1, t[0,1]=2
#  [5, 6],  # 第1行: 原始t[1,1]=5, t[1,2]=6
#  [9, 7]]  # 第2行: 原始t[2,2]=9, t[2,0]=7

# 示例1.2: 在行方向收集 (dim=0)
index2 = torch.tensor([
    [0, 1, 2],
    [2, 1, 0]
])

result2 = torch.gather(t, 0, index2)
# 第0行: 取t[0,0]=1, t[1,1]=5, t[2,2]=9
# 第1行: 取t[2,0]=7, t[1,1]=5, t[0,2]=3
# 结果: [[1, 5, 9], [7, 5, 3]]

"""
class ReplayBuffer:
    # 经验回放
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions) # 解包
        return np.array(state), action, reward, np.array(next_state), done
    
    def size(self):
        return len(self.buffer)
    

class Qnet(torch.nn.Module):
    # only 1 hidden layer
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

class DQN:
    def __init__(self, 
                 state_dim, hidden_dim, action_dim,
                 learning_rate, gamma, epsilon,
                 target_update, device):
        self.action_dim = action_dim

        self.q_net = Qnet(
            state_dim, hidden_dim,
            self.action_dim
        ).to(device)

        self.target_q_net = Qnet(
            state_dim, hidden_dim,
            self.action_dim
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(),
            lr=learning_rate
        )

        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = np.array(state) # .unsqueeze(0) # 保持批次维度
            state = torch.tensor([state], dtype=torch.float).to(self.device) ###
            action = self.q_net(state).argmax().item()
        return action
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)

        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1,1).to(self.device)
        
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        
        q_values = self.q_net(states).gather(1, actions) # 1 指列方向, actions是前边出来的值，是索引
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1,1) # 取列维度的max，转成列型对齐
        q_targets = rewards + self.gamma * max_next_q_values * (1-dones)

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict()
            )
        
        self.count += 1


if __name__ == '__main__':
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")
    
    env_name = 'CartPole-v0' # openai gym库 内置 注意版本，否则env返回后的state格式不兼容
    env = gym.make(env_name)
    random.seed(0)
    np.random.seed(0)
    env.seed(0) ###

    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim,
                lr, gamma, epsilon, target_update, device)
    
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d'% i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    # buffer里得积累点东西才能训这Q网络
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size) # b_d, d, done
                        # 这transition_dict就是一素材库
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict) # 这步就把经验加入了下次update的素材库了
                return_list.append(episode_return)
                if (i_episode + 1)%10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes/10*i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    # 到这儿，训练结束，开始画图
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    graph_dir = "./graph"
    import os
    os.makedirs(graph_dir, exist_ok=True)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}\n{}'.format(env_name, "带把儿的三蹦子，单轮儿"))

    fig1_path = os.path.join(graph_dir, f'dqn_{env_name}_returns.png')
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"已保存: {fig1_path}")

    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}\n{}'.format(env_name, "带把儿的三蹦子，移动平均"))

    fig2_path = os.path.join(graph_dir, f'dqn_{env_name}_moving_average.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"已保存: {fig2_path}")

    plt.show()
        