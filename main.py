# main.py
import gymnasium as gym # Use Gymnasium instead of gym
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import os
import ale_py

# Import from other files
from dqn_agent import DQNAgent, BATCH_SIZE  # Import BATCH_SIZE from dqn_agent.py
from debug_utils import print_tensor_info, visualize_state

plt.ion()  # Turn on interactive mode for real-time plotting

# --- Environment Setup ---
# Register ALE environments if not already registered
from gymnasium.envs.registration import register
try:
    # Try to import DemonAttack to see if it's already registered
    gym.make("ALE/DemonAttack-v5")
except gym.error.NameNotFound:
    # If not found, register ALE environments
    for game in ale_py.list_games():
        for dtype in ["", "-ram"]:
            register(
                id=f"ALE/{game}{dtype}-v5",
                entry_point="gymnasium.envs.atari:AtariEnv",
                kwargs={"game": game, "obs_type": "ram" if dtype == "-ram" else "rgb", "repeat_action_probability": 0.25},
                max_episode_steps=108000,
                nondeterministic=True,
            )

# Use the correct environment name
ENV_NAME = "ALE/DemonAttack-v5"  # Standard version

# --- Configuration / Hyperparameters ---
NUM_EPISODES = 2000      # Max number of training episodes
MAX_T = 10000           # Max number of timesteps per episode (optional)
PRINT_EVERY = 10        # How often to print avg score
SAVE_EVERY = 100        # How often to save the model checkpoint
LOG_DIR = "results"     # Directory for logs and plots
MODEL_DIR = "models"    # Directory for saved models
PLOT_EVERY = 50         # How often to update the training plot

# Create directories if they don't exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Environment Wrappers (Basic Example) ---
def make_env(env_id, seed=None, render_mode=None):
    # Create environment with frameskip=1 to disable internal frame skipping
    env = gym.make(env_id, frameskip=1, render_mode=render_mode)

    # Common wrappers based on DeepMind paper
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    # Manual preprocessing
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    
    # Use MaxAndSkip wrapper for frame skipping
    env = gym.wrappers.MaxAndSkipObservation(env, skip=4)
    
    # Use the renamed FrameStackObservation wrapper
    env = gym.wrappers.FrameStackObservation(env, 4)
    
    # Create a custom wrapper to transpose the observations to the format PyTorch expects
    class TransposeObservation(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            obs_shape = self.observation_space.shape
            # Change from (H, W, C) to (C, H, W)
            self.observation_space = gym.spaces.Box(
                low=0, 
                high=255, 
                shape=(obs_shape[2], obs_shape[0], obs_shape[1]),
                dtype=np.uint8
            )
        
        def observation(self, observation):
            # Transpose from (H, W, C) to (C, H, W)
            return np.transpose(observation, (2, 0, 1))
    
    # Uncomment this to use the transpose wrapper - currently we're fixing in replay buffer instead
    # env = TransposeObservation(env)

    # Print the observation shape for debugging
    test_obs, _ = env.reset()
    print(f"Observation shape after preprocessing: {test_obs.shape}")
    
    if seed is not None:
        env.reset(seed=seed)
    return env

# --- Main Training Loop ---
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment without rendering for training
    env = make_env(ENV_NAME)
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n
    print(f"State shape from space: {state_shape}, Number of actions: {num_actions}")
    
    # IMPORTANT: Use the state_shape directly since it's already in the correct format (4, 84, 84)
    # No need to transpose/correct it any further
    proper_input_shape = state_shape
    print(f"Input shape for model: {proper_input_shape}")
    
    # Create Agent with the correct shape
    agent = DQNAgent(input_shape=proper_input_shape, num_actions=num_actions, device=device)

    scores = []                         # list containing scores from each episode
    scores_window = deque(maxlen=100)   # last 100 scores for moving average
    eps_history = []                    # list containing epsilon values
    avg_scores = []                     # list containing average scores
    total_steps = 0

    start_time = time.time()

    # Create a figure for real-time plotting
    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    plt.suptitle('DQN Training Progress')
    plt.show(block=False)

    print("Starting Training...")
    for i_episode in range(1, NUM_EPISODES + 1):
        state, info = env.reset() # Gymnasium returns state, info
        score = 0
        episode_steps = 0
        episode_start = time.time()
        
        while True: # Loop until episode is done
            # Ensure state is in the correct shape
            if isinstance(state, np.ndarray) and len(state.shape) == 3:
                # Our state shape should be (4, 84, 84) - channels first
                if state.shape[0] != 4 and state.shape[2] == 4:
                    # If channels last, transpose to channels first
                    state = np.transpose(state, (2, 0, 1))
            
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated # Episode ends if terminated or truncated

            # Again ensure next_state is in the correct shape
            if isinstance(next_state, np.ndarray) and len(next_state.shape) == 3:
                if next_state.shape[0] != 4 and next_state.shape[2] == 4:
                    next_state = np.transpose(next_state, (2, 0, 1))

            # Store experience
            agent.memory.push(state, action, reward, next_state, done)

            # Learn only if enough samples and limit how often we call learn()
            if len(agent.memory) >= BATCH_SIZE:
                agent.learn()
            
            state = next_state
            score += reward
            total_steps += 1
            episode_steps += 1

            # Force end episodes after MAX_T steps to avoid infinite loops
            if done or episode_steps >= MAX_T:
                episode_time = time.time() - episode_start
                break

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps_history.append(agent.epsilon) # save epsilon value
        avg_scores.append(np.mean(scores_window)) # save average score

        # Print progress after EVERY episode
        print(f'Episode {i_episode}/{NUM_EPISODES} | Score: {score:.1f} | Avg: {np.mean(scores_window):.1f} | '
              f'Steps: {episode_steps} | Eps: {agent.epsilon:.2f} | Time: {episode_time:.1f}s')
        
        # Update the plots periodically
        if i_episode % PLOT_EVERY == 0:
            ax1.clear()
            ax2.clear()
            
            # Plot scores
            ax1.plot(scores, alpha=0.6, label='Score')
            ax1.plot(avg_scores, label='Avg Score (100 ep)')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Score')
            ax1.legend()
            ax1.set_title('Training Scores')
            
            # Plot epsilon
            ax2.plot(eps_history)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Epsilon')
            ax2.set_title('Exploration Rate')
            
            plt.tight_layout()
            plt.pause(0.1)  # Pause to update the plot
            
            # Save current data
            np.save(os.path.join(LOG_DIR, 'scores.npy'), np.array(scores))
            np.save(os.path.join(LOG_DIR, 'eps_history.npy'), np.array(eps_history))

        # Save model checkpoint
        if i_episode % SAVE_EVERY == 0:
            model_save_path = os.path.join(MODEL_DIR, f"demonattack_dqn_episode_{i_episode}.pth")
            agent.save(model_save_path)
            
            # Save the training plot
            plt.savefig(os.path.join(LOG_DIR, f'training_plot_episode_{i_episode}.png'))

    print("\nTraining finished.")
    env.close()

    # --- Plotting ---
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    # Plot moving average
    moving_avg = np.convolve(scores, np.ones(100)/100, mode='valid')
    plt.plot(np.arange(len(moving_avg)) + 99, moving_avg, label='Moving Avg (100 episodes)')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title(f'DQN Training Scores - {ENV_NAME}')
    plt.legend()
    plot_save_path = os.path.join(LOG_DIR, f"dqn_training_scores_{ENV_NAME.replace('/', '_')}.png")
    plt.savefig(plot_save_path)
    print(f"Score plot saved to {plot_save_path}")
    # plt.show() # Uncomment to display plot immediately