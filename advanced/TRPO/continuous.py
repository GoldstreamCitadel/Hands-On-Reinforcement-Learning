import torch
import torch.nn.functional as F
from discrete import ValueNet

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
        # softplus(x) = log(1 + exp(x)) 确保有效能导可反传
        return mu, std # 高斯分布的均值和标准差
    

class TRPOContinuous:
    def __init__(self, hidden_dim, state_space, action_space,
                 lmbda, kl_constraint, alpha, critic_lr,
                 gamma, device):
        state_dim = state_space.shape[0]
        action_dim = action_space.shape[0]

        self.actor = PolicyNetContinuous(state_dim, hidden_dim,
                                         action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.lmbda = lmbda
        self.kl_constraint = kl_constraint
        self.alpha = alpha
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, std = self.actor(state)
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample()
        return [action.item()]
    
    def hessian_matrix_vector_product(self,
                                      states,
                                      old_action_dists,
                                      vector,
                                      damping=0.1):
        mu, std = self.actor(states)
        new_action_dists = torch.distributions.Nromal(mu, std)
        kl = torch.mean(
            torch.distributions.kl.kl_divergence(
                old_action_dists, new_action_dists))
        # kl对策略网络参数求导
        kl_grad = torch.autograd.grad(
            kl, self.actor.parameters(),
            create_praph = True
        )
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)

        grad2 = torch.autograd.grad(
            kl_grad_vector_product, self.actor.parameters()
        )
        grad2_vector = torch.cat(
           [grad.contiguous.view(-1) for grad in grad2] 
        )
        # 想象在山上找最低点：
        #F告诉你山的曲率​（哪个方向陡，哪个方向缓）
        # 自然梯度 F^-1 * g是“考虑地形”的最佳下降方向
        #但有些方向曲率几乎为0（平地）→ F几乎奇异 
        # → 自然梯度在这些方向会爆炸S
        #阻尼项确保你不会在平地上“瞬移”到无限远
        return grad2_vector + damping * vector
    
    def conjugate_gradient(self, grad, states, old_action_dists):
        pass

    def compute_surrogate_obj(self, states, actions,
                              old_log_probs, actor):
        pass

    def line_search(self, states, actions, advantage,
                    old_log_probs, old_action_dists,
                    max_vec):
        pass

    def policy_learn(self, states, actions,
                     old_action_dists, old_log_probs,
                     advantage):
        pass

    def update(self, transition_dict):
        pass


if __name__ == '__main__':
    pass