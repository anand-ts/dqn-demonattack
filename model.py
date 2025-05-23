# model.py
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import random

# NoisyLinear layer for parameterized exploration (NoisyNet)
class NoisyLinear(nn.Module):
    """Noisy linear module with factorized gaussian noise."""
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        # Buffers for noise
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        # Outer product for factorized noise
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)

class QNetwork(nn.Module):
    """Dueling DQN Network."""

    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        channels = input_shape[0]
        print(f"Using {channels} input channels (Dueling DQN)")

        # CNN layers
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        # Calculate the flattened size after conv layers
        dummy_input = torch.zeros(1, channels, input_shape[1], input_shape[2])
        with torch.no_grad():
            o = self.conv1(dummy_input)
            o = self.conv2(o)
        conv_out_size = int(np.prod(o.size()))

        # Dueling streams
        self.fc_adv = nn.Linear(conv_out_size, 256)
        self.fc_val = nn.Linear(conv_out_size, 256)
        self.advantage = nn.Linear(256, num_actions)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        # Input normalization
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        debug_print = random.random() < 0.0001
        if debug_print:
            print(f"Model input shape: {x.shape}")

        if x.dim() != 4:
            print(f"WARNING: Unexpected input dimensions: {x.shape}. Expected 4 dimensions (B, C, H, W).")
            if x.dim() == 3 and x.shape[0] == self.input_shape[0] and x.shape[1] == self.input_shape[1] and x.shape[2] == self.input_shape[2]:
                x = x.unsqueeze(0)
                if debug_print: print(f"Reshaped to: {x.shape}")

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)

        adv = F.relu(self.fc_adv(x))
        val = F.relu(self.fc_val(x))
        adv = self.advantage(adv)
        val = self.value(val)
        # Combine streams: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_values = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_values

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()