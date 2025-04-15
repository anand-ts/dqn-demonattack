# replay_buffer.py
import random
import collections
import numpy as np
import torch

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            capacity (int): maximum size of buffer
            device (torch.device): device to store tensors on (cpu or cuda)
        """
        self.memory = collections.deque(maxlen=capacity)
        self.capacity = capacity
        self.device = device

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # Ensure data is in a basic format (e.g., numpy array for states)
        # Gymnasium usually returns numpy arrays for observations
        e = (state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)
        
        # Remove debug prints to reduce console spam
        # Only print debug info occasionally (0.1% of the time)
        debug_print = random.random() < 0.001

        # Extract and process states
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for e in experiences:
            if e is not None:
                state, action, reward, next_state, done = e
                
                # Ensure state is in the right format (C, H, W)
                if state.shape[0] != 4 and state.shape[2] == 4:
                    state = np.transpose(state, (2, 0, 1))
                    next_state = np.transpose(next_state, (2, 0, 1))
                
                states.append(state)
                actions.append([action])  # Make action a single-item list
                rewards.append([reward])  # Make reward a single-item list
                next_states.append(next_state)
                dones.append([float(done)])  # Convert boolean to float
        
        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Only print debug info occasionally
        if debug_print:
            print(f"Batch states shape: {states.shape}")
        
        # Convert to PyTorch tensors
        states_t = torch.from_numpy(states).float().to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().to(self.device)

        return (states_t, actions_t, rewards_t, next_states_t, dones_t)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)