# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math

# Import from other files in the same directory
from model import QNetwork
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from collections import deque

# --- Hyperparameters ---
BUFFER_SIZE = int(1e5)  # memory capacity changed to 1e5 (was 1e6)
BATCH_SIZE = 32         # batch size for gradient updates (as per original DQN)
GAMMA = 0.99            # discount factor
LR = 2.5e-4             # learning rate (0.00025 as per original)
FRAME_SKIP_FACTOR = 4   # Define this based on MaxAndSkipObservation setting
TARGET_UPDATE_FREQ = 10000 // FRAME_SKIP_FACTOR  # Update every 2500 agent steps (10K frames)
N_STEPS = 3             # multi-step return length (set to 3 for n-step returns)
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 1000000 // FRAME_SKIP_FACTOR  # Decay over 250K agent steps (1M frames)
USE_PRIORITIZED_REPLAY = True  # ENABLE prioritized replay for PER
USE_DOUBLE_DQN = True   # enable Double DQN algorithm

class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, input_shape, num_actions, device, use_double_dqn=USE_DOUBLE_DQN):
        """Initialize an Agent object.

        Params
        ======
            input_shape (tuple): dimension of each state (e.g., (4, 84, 84))
            num_actions (int): number of possible actions
            device (torch.device): device to use (cpu or cuda)
            use_double_dqn (bool): whether to use Double DQN algorithm
        """
        self.device = device
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.use_double_dqn = use_double_dqn

        # Q-Network
        self.policy_net = QNetwork(input_shape, num_actions).to(device)
        self.target_net = QNetwork(input_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is only for inference

        # Original DQN uses RMSprop optimizer
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=LR, alpha=0.95, eps=0.01)
        
        # Replay memory
        if USE_PRIORITIZED_REPLAY:
            self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, device, input_shape)
            self.prioritized = True
        else:
            self.memory = ReplayBuffer(BUFFER_SIZE, device)
            self.prioritized = False

        # Initialize step counters: t_step for target updates, frame_idx for epsilon annealing
        self.t_step = 0
        self.frame_idx = 0
        # Epsilon value
        self.epsilon = EPSILON_START
        # n-step buffer for multi-step returns (not used if N_STEPS=1)
        self.n_steps = N_STEPS
        self.gamma = GAMMA
        if self.n_steps > 1:
            self.n_step_buffer = deque(maxlen=self.n_steps)
        else:
            self.n_step_buffer = None # Not needed for 1-step
        # Track losses for monitoring
        self.losses = []

    def select_action(self, state):
        """Returns actions for given state as per current policy (epsilon-greedy)."""
        # Increment both our agent decision counter for target updates
        # and our frame counter for epsilon decay
        self.t_step += 1
        self.frame_idx += 1
        
        # Epsilon-greedy action selection
        sample = random.random()
        
        # Linear decay for epsilon based on frame index
        self.epsilon = max(EPSILON_END,
                          EPSILON_START - (EPSILON_START - EPSILON_END) * self.frame_idx / EPSILON_DECAY)

        if sample > self.epsilon:
            # Exploit: select the best action
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            
            # Ensure state has the right shape
            if len(state.shape) == 4 and state.shape[1] != self.input_shape[0]:
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
            return

        # Learning step (noisy nets removed)
        # Sample from replay buffer
        sample_output = self.memory.sample(BATCH_SIZE)
        if isinstance(sample_output, (tuple, list)) and len(sample_output) == 7:
            states, actions, rewards, next_states, dones, weights, indices = sample_output
        elif isinstance(sample_output, (tuple, list)) and len(sample_output) == 5:
            states, actions, rewards, next_states, dones = sample_output
            weights = torch.ones_like(rewards)
            indices = None
        else:
            raise ValueError("Unexpected output from replay buffer sample")

        with torch.no_grad():
            if self.use_double_dqn:
                # --- Double DQN Target Calculation ---
                # Get actions from policy network
                next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
                # Get Q-values from target network for those actions
                next_q_values = self.target_net(next_states).gather(1, next_actions)
            else:
                # --- Original DQN Target Calculation ---
                # Get the maximum predicted Q value for the next states from the target network
                next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
                
            # Compute Q targets for current states
            q_targets = rewards + (self.gamma * next_q_values * (1 - dones))

        # Get expected Q values from policy model
        q_expected = self.policy_net(states).gather(1, actions)

        # Compute element‐wise TD error
        td_errors = q_targets - q_expected
        # Mean Squared Error loss
        element_loss = F.mse_loss(q_expected, q_targets, reduction='none')
        loss = (weights * element_loss).mean()
        
        # Store loss for monitoring
        self.losses.append(loss.item())
        
        # Clear gradients and perform optimization step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update priorities in PER if enabled
        if hasattr(self.memory, "update_priorities") and indices is not None:
            # Use absolute TD errors as priorities
            new_priorities = td_errors.detach().abs().cpu().numpy().flatten()
            self.memory.update_priorities(indices, new_priorities)  # type: ignore[attr-defined]

        # Update target network by hard copy every TARGET_UPDATE_FREQ steps
        if self.t_step % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Target network updated at step {self.t_step}. Training is {(self.frame_idx / EPSILON_DECAY) * 100:.1f}% complete.")

    def _update_target_network(self):
         # Deprecated soft update (unused)
         pass

    def save(self, filename):
        """Save the policy network weights."""
        checkpoint = {
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'epsilon': self.epsilon,
            't_step': self.t_step,
            'frame_idx': self.frame_idx,
            'use_double_dqn': self.use_double_dqn
        }
        torch.save(checkpoint, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        """Load the policy network weights."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'losses' in checkpoint:
            self.losses = checkpoint['losses']
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']
        if 't_step' in checkpoint:
            self.t_step = checkpoint['t_step']
        if 'frame_idx' in checkpoint:
            self.frame_idx = checkpoint['frame_idx']
        if 'use_double_dqn' in checkpoint:
            self.use_double_dqn = checkpoint['use_double_dqn']
        
        self.policy_net.eval()
        self.target_net.eval()
        print(f"Model loaded from {filename}")
    def _get_n_step_info(self):
        """Compute n-step return and terminal info from buffer."""
        if self.n_steps == 1: # Should not be called if n_steps is 1
            raise ValueError("N-step info should not be called when N_STEPS is 1.")
        
        # Ensure n_step_buffer exists and is not None
        if self.n_step_buffer is None:
            raise ValueError("n_step_buffer is None but _get_n_step_info was called")
            
        reward, next_state, done = 0.0, None, False
        
        # Now safe to iterate over n_step_buffer
        for idx, (s, a, r, ns, d) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** idx) * r
            if d:
                next_state = ns
                done = True
                break
            next_state = ns
        return reward, next_state, done

    def step(self, state, action, reward, next_state, done):
        """Process a step: add to replay buffer using n-step returns if enabled."""
        if self.n_steps == 1 or self.n_step_buffer is None:
            self.memory.push(state, action, reward, next_state, done)
        else:
            # Add transition to n-step buffer
            self.n_step_buffer.append((state, action, reward, next_state, done))
            # If buffer is full or episode ends, push n-step transition
            if len(self.n_step_buffer) == self.n_steps or done:
                reward_n, next_state_n, done_n = self._get_n_step_info()
                state_0, action_0, _, _, _ = self.n_step_buffer[0]
                self.memory.push(state_0, action_0, reward_n, next_state_n, done_n)
            # If episode ends, clear the buffer
            if done:
                self.n_step_buffer.clear()