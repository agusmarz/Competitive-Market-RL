import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # orthogonal init, standard for PPO actor/critic nets
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorNN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            layer_init(nn.Linear(in_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)),     nn.Tanh(),
        )
        self.mu_head = layer_init(nn.Linear(64, out_dim), std=0.01)

        # state-independent log std, starts at 0
        self.log_std = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        h   = self.shared(x)
        mu  = self.mu_head(h)
        std = self.log_std.expand_as(mu).exp() # no clamp on purpose
        return mu, std


class CriticNN(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(in_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)),     nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        return self.net(x)


class RunningMeanStd:
    # tracks a running mean/var, used to normalize obs and rewards on the fly
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4

    def update(self, x):
        # parallel variance formula, so we don't need to keep every sample around
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count
