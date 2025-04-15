# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

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

        # CNN layers - using the correct number of input channels
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the flattened size
        dummy_input = torch.zeros(1, channels, input_shape[1], input_shape[2])
        print(f"Dummy input shape: {dummy_input.shape}")
        conv_out_size = self._get_conv_out(dummy_input)
        print(f"Conv output size: {conv_out_size}")

        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _get_conv_out(self, shape):
        """Helper function to calculate the output size of convolutional layers."""
        o = self.conv1(shape)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        """Build a network that maps state -> action values."""
        # Input normalization
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        
        # Only print debug info occasionally (0.1% of the time)
        debug_print = random.random() < 0.001
        if debug_print:
            print(f"Model input shape: {x.shape}")
        
        # Handle dimensionality issues
        if x.dim() > 4:  # More dimensions than expected
            # Reshape to correct dimensionality
            print(f"WARNING: Input has too many dimensions: {x.shape}")
            batch_size = 1 if x.shape[0] == 1 else x.shape[0] // (84*84)
            print(f"Attempting reshape with batch_size={batch_size}")
            x = x.reshape(batch_size, 4, 84, 84)
            print(f"Reshaped to: {x.shape}")
        
        # Apply convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)

        return q_values