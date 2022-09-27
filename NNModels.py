from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class CriticNet(nn.Module):
    def __init__(self, state_dim, lr):
        super(CriticNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )
        self.optimizer = Adam(self.net.parameters(), lr=lr)

    def forward(self, state):
        value = self.net(state)
        return value


class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, lr):
        super(ActorNet, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ELU(),
        )
        self.mean_layer = nn.Linear(32, action_dim)
        self.std_layer = nn.Linear(32, action_dim)

        all_params = list(self.shared_layers.parameters())\
                     + list(self.mean_layer.parameters())\
                     + list(self.std_layer.parameters())
        self.optimizer = Adam(all_params, lr=lr)

    def forward(self, state):
        shared_pass = self.shared_layers(state)
        mean = self.mean_layer(shared_pass)
        std_pass = self.std_layer(shared_pass)
        std = F.softplus(std_pass) + 1e-5
        distribution = D.normal.Normal(mean, std)
        return distribution
