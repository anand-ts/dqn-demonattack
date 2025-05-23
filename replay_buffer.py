# replay_buffer.py
import random
import collections
import numpy as np
import torch
from collections import deque, namedtuple

# Define a named tuple for storing experiences
Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            capacity (int): maximum size of buffer
            device (torch.device): device to store tensors on (cpu or cuda)
        """
        self.memory = deque(maxlen=capacity)
        self.device = device
        self.batch_states = None
        self.batch_next_states = None

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)
        
        # Find a valid state to determine shape
        valid_state = None
        for exp in experiences:
            if exp.state is not None:
                valid_state = exp.state
                break
                
        if valid_state is None:
            raise ValueError("No valid states in sampled batch")
        
        state_shape = valid_state.shape
        
        # Create fresh tensors for this batch to avoid None issues
        batch_states = torch.zeros((batch_size, *state_shape), dtype=torch.float32, device=self.device)
        batch_next_states = torch.zeros((batch_size, *state_shape), dtype=torch.float32, device=self.device)
        actions = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        rewards = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
        dones = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
        
        for i, e in enumerate(experiences):
            # Copy data into tensors
            if e.state is not None:
                batch_states[i] = torch.as_tensor(e.state, dtype=torch.float32)
            actions[i] = torch.tensor(e.action, dtype=torch.long)
            rewards[i] = torch.tensor(e.reward, dtype=torch.float32)
            if e.next_state is not None:
                batch_next_states[i] = torch.as_tensor(e.next_state, dtype=torch.float32)
            dones[i] = torch.tensor(e.done, dtype=torch.float32)
        
        # Save for potential reuse
        self.batch_states = batch_states
        self.batch_next_states = batch_next_states
        
        return (batch_states, actions, rewards, batch_next_states, dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class PrioritizedReplayBuffer:
    """Prioritized Replay Buffer to store experiences and sample based on TD error."""
    
    def __init__(self, buffer_size, device, state_shape, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=100000):
        """Initialize the Prioritized Replay Buffer.
        
        Args:
            buffer_size: Maximum size of the buffer
            device: PyTorch device
            state_shape: Shape of the state observations
            alpha: How much prioritization to use (0 = no prioritization, 1 = full prioritization)
            beta_start: Initial value of beta for importance-sampling weights
            beta_end: Final value of beta
            beta_frames: Number of frames over which to anneal beta from beta_start to beta_end
        """
        self.buffer_size = buffer_size
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.frame = 1  # Tracks number of items pushed, for beta annealing
        self.size = 0
        self.next_idx = 0
        
        # Initialize buffers with proper shapes
        self.states = np.zeros((buffer_size, *state_shape), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, *state_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, 1), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.uint8)
        
        # Initialize prioritized replay variables
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.pos = 0
        self.size = 0
    
    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory with maximum priority."""
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        # Store experience components
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        
        # Set max priority for new experience
        self.priorities[self.pos] = max_priority
        
        # Update position and size
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
        self.frame += 1 # frame should be incremented here for beta annealing
    
    def sample(self, batch_size):
        """Sample a batch of experiences based on their priorities."""
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer")
            
        # Calculate current beta value based on number of frames/experiences added
        # self.frame is incremented by the agent after each push to memory
        beta_progress = min(1.0, self.frame / self.beta_frames)
        beta = self.beta_start + beta_progress * (self.beta_end - self.beta_start)
        
        # Calculate sampling probabilities
        # Ensure we only sample from filled parts of the buffer
        current_priorities = self.priorities[:self.size]
        
        if current_priorities.sum() == 0: # Avoid division by zero if all priorities are zero
            # Fallback to uniform sampling if all priorities are zero
            # This can happen if td_errors are consistently zero, which is unlikely with epsilon > 0
            indices = np.random.choice(self.size, batch_size, replace=True) # replace=True if batch_size > self.size
            probabilities_for_indices = np.full(batch_size, 1.0 / self.size)
        else:
            probabilities = current_priorities ** self.alpha
            probabilities /= probabilities.sum()
            indices = np.random.choice(self.size, batch_size, p=probabilities, replace=True) # replace=True if batch_size > self.size
            probabilities_for_indices = probabilities[indices]

        # Calculate importance-sampling weights
        weights = (self.size * probabilities_for_indices) ** (-beta)
        weights /= weights.max()  # Normalize to have max weight = 1
        weights = np.array(weights, dtype=np.float32).reshape(-1, 1)
        
        # Convert to torch tensors
        states = torch.from_numpy(self.states[indices]).float().to(self.device)
        actions = torch.from_numpy(self.actions[indices]).long().to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).float().to(self.device)
        next_states = torch.from_numpy(self.next_states[indices]).float().to(self.device)
        dones = torch.from_numpy(self.dones[indices]).float().to(self.device)
        weights = torch.from_numpy(weights).float().to(self.device)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled indices."""
        for idx, priority in zip(indices, priorities):
            # Ensure priority is positive and add a small epsilon for stability
            self.priorities[idx] = abs(priority) + 1e-6 # priorities should be based on abs TD error
            
    def __len__(self):
        """Return the current size of the buffer."""
        return self.size