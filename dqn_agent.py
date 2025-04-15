# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

# Import from other files in the same directory
from model import QNetwork
from replay_buffer import ReplayBuffer

# --- Hyperparameters ---
# You might move these to a config.py or pass them to __init__
BUFFER_SIZE = int(1e5)  # Replay buffer size (e.g., 100k or 1M)
BATCH_SIZE = 64         # Minibatch size
GAMMA = 0.99            # Discount factor
LR = 1e-4               # Learning rate
TARGET_UPDATE_FREQ = 1000 # How often to update the target network (steps)
# Epsilon parameters (adjust these)
EPSILON_START = 1.0
EPSILON_END = 0.02 # Or 0.1 / 0.01
EPSILON_DECAY = 30000 # How many steps to decay epsilon over

class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, input_shape, num_actions, device):
        """Initialize an Agent object.

        Params
        ======
            input_shape (tuple): dimension of each state (e.g., (4, 84, 84))
            num_actions (int): number of possible actions
            device (torch.device): device to use (cpu or cuda)
        """
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.device = device

        # Q-Network
        self.policy_net = QNetwork(input_shape, num_actions).to(device)
        self.target_net = QNetwork(input_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is only for inference

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR) # Adam often works well

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, device)

        # Initialize time step (for updating every TARGET_UPDATE_FREQ steps)
        self.t_step = 0
        # Epsilon value
        self.epsilon = EPSILON_START


    def select_action(self, state):
        """Returns actions for given state as per current policy (epsilon-greedy)."""
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            # Exploit: select the best action
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            
            # Ensure state has the right shape
            if len(state.shape) == 4 and state.shape[1] != self.input_shape[0]:
                # If channels not in the right dimension, transpose
                state = state.permute(0, 3, 1, 2)
                
            self.policy_net.eval()
            with torch.no_grad():
                action_values = self.policy_net(state)
            self.policy_net.train()
            action = np.argmax(action_values.cpu().data.numpy())
            return action
        else:
            # Explore: select a random action
            return random.choice(np.arange(self.num_actions))

    def learn(self):
        """Update value parameters using given batch of experience tuples."""
        if len(self.memory) < BATCH_SIZE:
            return # Not enough samples yet

        # Only sample and learn occasionally based on step count
        self.t_step += 1
        
        # Only do the actual learning every 4 steps to reduce computation
        if self.t_step % 4 != 0:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # --- Calculate Target Q Values ---
        # Get max predicted Q values for next states from target model
        with torch.no_grad(): # We don't need gradients for target calculation
             # Use target_net for stability
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        # target = reward + gamma * max_next_Q * (1 - done)
        q_targets = rewards + (GAMMA * next_q_values * (1 - dones))

        # --- Calculate Expected Q Values ---
        # Get expected Q values from policy model for the actions taken
        q_expected = self.policy_net(states).gather(1, actions)

        # --- Compute Loss ---
        # loss = F.mse_loss(q_expected, q_targets)
        loss = F.smooth_l1_loss(q_expected, q_targets) # Huber loss - often more robust

        # --- Optimize the Model ---
        self.optimizer.zero_grad() # Clear old gradients
        loss.backward()           # Calculate gradients
        # Optional: Gradient clipping (prevents exploding gradients)
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100) # Or clip_grad_norm_
        self.optimizer.step()     # Update weights

        # --- Update Target Network ---
        # soft update is another option: tau*policy_net + (1-tau)*target_net
        if self.t_step % TARGET_UPDATE_FREQ == 0:
            self._update_target_network()

        # --- Update Epsilon ---
        self.epsilon = max(EPSILON_END, self.epsilon - (EPSILON_START - EPSILON_END) / EPSILON_DECAY)
        # Or use exponential decay or other schedules

    def _update_target_network(self):
         """Hard update: Copy weights from policy_net to target_net."""
         print(f"Updating target network. Epsilon: {self.epsilon:.4f}")
         self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename):
        """Save the policy network weights."""
        torch.save(self.policy_net.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        """Load the policy network weights."""
        self.policy_net.load_state_dict(torch.load(filename, map_location=self.device))
        # Also copy to target network if loading for further training/evaluation
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval() # Set to eval mode if loading just for inference
        self.target_net.eval()
        print(f"Model loaded from {filename}")