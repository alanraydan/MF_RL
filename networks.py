import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class CriticNet(nn.Module):
    def __init__(self, state_dim):
        super(CriticNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        value = self.net(state)
        return value


class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNet, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
        )
        self.mean_layer = nn.Linear(64, action_dim)
        self.std_layer = nn.Linear(64, action_dim)

    def forward(self, state):
        shared_pass = self.shared_layer(state)
        mean = self.mean_layer(shared_pass)
        std_pass = self.std_layer(shared_pass)
        std = F.softplus(std_pass) + 1e-5
        distribution = D.normal.Normal(mean, std)
        return distribution
    

class MultiCriticNet(nn.Module):
    def __init__(self, state_dim, n_critics):
        super(MultiCriticNet, self).__init__()
        # Create a list of `n_critics` `CriticNet`s
        self.critics = nn.ModuleList(
            [CriticNet(state_dim) for _ in range(n_critics)]
        )

    def forward(self, state, critic_idx):
        return self.critics[critic_idx](state)


class MultiActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, n_actors):
        super(MultiActorNet, self).__init__()
        # Create a list of `n_actors` `ActorNet`s
        self.actors = nn.ModuleList(
            [ActorNet(state_dim, action_dim) for _ in range(n_actors)]
        )

    def forward(self, state, actor_idx):
        return self.actors[actor_idx](state)
