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
    """Actor (Policy) Model."""

    def __init__(self, input_shape, num_actions):
        """Initialize parameters and build model.
        Params
        ======
            input_shape (tuple): Shape of the observation space (e.g., (4, 84, 84))
            num_actions (int): Number of possible actions
        """
        super(QNetwork, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # For frame-stacked Atari, we expect channels to be 4
        # The first dimension should be the channel dimension
        channels = input_shape[0]
        print(f"Using {channels} input channels")

        # CNN layers - following original DQN architecture
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        # Calculate the flattened size after conv layers
        dummy_input = torch.zeros(1, channels, input_shape[1], input_shape[2])
        with torch.no_grad():
            o = self.conv1(dummy_input)
            o = self.conv2(o)
        conv_out_size = int(np.prod(o.size()))

        # Fully connected layers and output (original DQN)
        self.fc = nn.Linear(conv_out_size, 256)
        self.out = nn.Linear(256, num_actions)

    # Removed _get_conv_out as unnecessary

    def forward(self, x):
        """Build a network that maps state -> action values."""
        # Input normalization
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        
        # Only print debug info occasionally (0.1% of the time)
        debug_print = random.random() < 0.0001 # Reduced frequency
        if debug_print:
            print(f"Model input shape: {x.shape}")
        
        # Handle dimensionality issues - REMOVED complex reshaping.
        # The input x is expected to be (batch_size, channels, height, width)
        # e.g., (64, 4, 84, 84) or (1, 4, 84, 84)
        if x.dim() != 4:
            print(f"WARNING: Unexpected input dimensions: {x.shape}. Expected 4 dimensions (B, C, H, W).")
            # Attempt a sensible reshape if it's a single unbatched observation (C, H, W)
            if x.dim() == 3 and x.shape[0] == self.input_shape[0] and x.shape[1] == self.input_shape[1] and x.shape[2] == self.input_shape[2]:
                x = x.unsqueeze(0) # Add batch dimension
                if debug_print: print(f"Reshaped to: {x.shape}")
            # else: # If other unexpected shapes, it might be better to raise an error or log more verbosely.
                # For now, we let it pass, and PyTorch will likely raise an error in conv layers if shape is incompatible.

        # Apply convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        # Fully connected and output
        x = F.relu(self.fc(x))
        q_values = self.out(x)
        return q_values
   
    def reset_noise(self):
        """Reset noise in all NoisyLinear layers."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()