import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from DQN import rl_utils
from ppo_dis import ValueNet


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std

class PPOContinuous:
    def __init__(
        self, 
        state_dim, hidden_dim, action_dim,
        actor_lr, critic_lr, lmbda, epochs,
        eps, gamma, device
        ):
        self.actor = PolicyNetContinuous(
            state_dim,
            hidden_dim,
            action_dim
        ).to(device)
        self.critic = ValueNet(
            state_dim,
            hidden_dim
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr = actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr = critic_lr
        )
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor(
            [state],
            dtype=torch.float
        ).to(self.device)
        # 下面这句话不是传参初始化
        # 而是输入，只要tensor形状和第一层相符,
        # 那就是对的
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(
            mu, sigma
        )
        action = action_dist.sample()
        return [action.item()] # 输出可看作一个数，程度大小

    def update(self, transition_dict):
        states = torch.tensor(
            transition_dict['states'],
            dtype=torch.float
        ).to(self.device)
        actions = torch.tensor(
            transition_dict['actions'],
            dtype = torch.float
        ).view(-1,1).to(self.device)
        rewards = torch.tensor(
            transition_dict['rewards'],
            dtype = torch.float
        ).view(-1,1).to(self.device)
        next_states = torch.tensor(
            transition_dict['next_states'],
            dtype = torch.float
        ).to(self.device)
        dones = torch.tensor(
            transition_dict['dones'],
            dtype = torch.float
        ).view(-1,1).to(self.device)

        # like TRPO 修改奖励 方便训练
        rewards = (rewards+8.0)/8.0
        td_target = rewards + self.gamma * \
            self.critic(next_states)*(1-dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(
            self.gamma,
            self.lmbda,
            td_delta.cpu()
        ).to(self.device)

        mu, std = self.actor(states)
        # 老的mu std不反传，数学正确
        action_dists = torch.distributions.Normal(
            mu.detach(),
            std.detach()
        )
        # 动作是正态分布
        # 进去batch个actions值 出来一堆概率值
        old_log_probs = action_dists.log_prob(actions)
        """
        actions是什么？
        这是之前智能体在环境中实际执行的动作，
        存储在经验回放缓冲区中
        ​不是索引，而是具体的动作数值
        例如在机器人控制中，
        可能是[0.5, -0.3, 1.2]这样的关节扭矩值
        """
        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            # 采样比率
            ratio = torch.exp(log_probs - old_log_probs)

            surr1 = ratio * advantage
            # 比如ratio算出来可能是1.5
            # 我给它限制1.2封顶
            # 每次变化幅度就更合适
            surr2 = torch.clamp(
                ratio,
                1-self.eps,
                1+self.eps
            ) * advantage

            actor_loss = torch.mean(
                -torch.min(surr1, surr2)
            )
            critic_loss = torch.mean(
                F.mse_loss(
                    self.critic(states),
                    td_target.detach()
                )
            )

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


if __name__ == '__main__':
    actor_lr = 1e-4
    critic_lr = 5e-3
    num_episodes = 2000
    hidden_dim = 128
    gamma = 0.9
    lmbda = 0.9
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]  # 连续动作空间
    agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                        lmbda, epochs, eps, gamma, device)

    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

    import matplotlib.pyplot as plt
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 21)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()
