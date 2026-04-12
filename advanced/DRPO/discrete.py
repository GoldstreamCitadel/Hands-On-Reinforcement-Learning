import gym
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
from DQN import rl_utils


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DRPO:
    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr, lmbda, epochs, eps,
                 gamma, target_kl, beta, entropy_coef, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.target_kl = target_kl
        self.beta = beta
        self.entropy_coef = entropy_coef
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                        dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)

        old_probs = self.actor(states).detach()
        old_action_dist = torch.distributions.Categorical(old_probs)
        old_log_probs = torch.log(old_probs.gather(1, actions) + 1e-8).detach()

        kl = torch.tensor(0.0, device=self.device)
        for _ in range(self.epochs):
            probs = self.actor(states)
            action_dist = torch.distributions.Categorical(probs)
            log_probs = torch.log(probs.gather(1, actions) + 1e-8)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            kl = torch.mean(
                torch.distributions.kl.kl_divergence(old_action_dist,
                                                     action_dist))
            entropy = torch.mean(action_dist.entropy())

            actor_loss = torch.mean(-torch.min(surr1, surr2))
            actor_loss = actor_loss + self.beta * kl - self.entropy_coef * entropy
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        if kl.item() > 1.5 * self.target_kl:
            self.beta = min(self.beta * 2.0, 100.0)
        elif kl.item() < self.target_kl / 1.5:
            self.beta = max(self.beta / 2.0, 1e-4)


if __name__ == '__main__':
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    target_kl = 0.01
    beta = 1.0
    entropy_coef = 0.01
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DRPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                 epochs, eps, gamma, target_kl, beta, entropy_coef, device)
    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DRPO on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DRPO on {}'.format(env_name))
    plt.show()
