import gymnasium as gym # Use Gymnasium instead of gym
import sys
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import os
import ale_py

LOG_DIR = "results"     # Directory for logs and plots
MODEL_DIR = "models"    # Directory for saved models
EPISODE_METRICS_CSV = os.path.join(LOG_DIR, "episode_metrics.csv")
# Placeholder for TensorBoard writer; initialized only when running training
writer = None  # type: ignore

# --- Log all stdout/stderr to a file as well as terminal ---
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

from utils import log_training_stats


# Import from other files
from dqn_agent import DQNAgent, BATCH_SIZE  # Import BATCH_SIZE from dqn_agent.py
from debug_utils import print_tensor_info, visualize_state

# plt.ion()  # Turn on interactive mode for real-time plotting

# --- Environment Setup ---
# Register ALE environments if not already registered
from gymnasium.envs.registration import register
try:
    # Try to import DemonAttack to see if it's already registered
    gym.make("ALE/DemonAttack-v5")
except gym.error.NameNotFound:
    # If not found, register the DemonAttack environment manually
    register(
        id="ALE/DemonAttack-v5",
        entry_point="gymnasium.envs.atari:AtariEnv",
        kwargs={"game": "demon_attack", "obs_type": "rgb", "repeat_action_probability": 0.25},
        max_episode_steps=108000,
        nondeterministic=True,
    )
    register(
        id="ALE/DemonAttack-ram-v5",
        entry_point="gymnasium.envs.atari:AtariEnv",
        kwargs={"game": "demon_attack", "obs_type": "ram", "repeat_action_probability": 0.25},
        max_episode_steps=108000,
        nondeterministic=True,
    )

# Use the correct environment name
ENV_NAME = "ALE/DemonAttack-v5"  # Standard version

# --- Configuration / Hyperparameters ---
# NUM_EPISODES = 100      # Old: Limited episodes approach
TOTAL_FRAMES_TO_TRAIN = 1000000  # Start with 1M frames (paper used 10M)
MAX_T = 10000           # Max number of timesteps per episode
PRINT_EVERY = 1         # Print every episode
SAVE_EVERY = 100        # Save every 100 episodes (less frequent for long training)
PLOT_EVERY = 100        # Plot less frequently for long training
LOG_DIR = "results"     # Directory for logs and plots
MODEL_DIR = "models"    # Directory for saved models

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
    
    # Frame skipping as per original DQN paper (k=4)
    env = gym.wrappers.MaxAndSkipObservation(env, skip=4)  # Changed back to 4 as per original paper
    
    # Use the renamed FrameStackObservation wrapper
    env = gym.wrappers.FrameStackObservation(env, 4)
    
    # Create a custom wrapper to transpose the observations to the format PyTorch expects
    class TransposeObservation(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            obs_shape = self.observation_space.shape
            # Fix for shape access - handle potential None value
            if obs_shape is not None and len(obs_shape) >= 3:
                # Change from (H, W, C) to (C, H, W)
                self.observation_space = gym.spaces.Box(
                    low=0, 
                    high=255, 
                    shape=(obs_shape[2], obs_shape[0], obs_shape[1]),
                    dtype=np.uint8
                )
            else:
                print("Warning: Unexpected observation shape:", obs_shape)
        
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

import argparse

if __name__ == "__main__":
    # Defer heavy/optional imports to runtime so visualize.py can import from this module
    from torch.utils.tensorboard.writer import SummaryWriter
    # Ensure directories exist and set up logging/metrics files
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Mirror stdout/stderr to file
    console_log_path = os.path.join(LOG_DIR, "console.log")
    sys.stdout = Tee(sys.stdout, open(console_log_path, "a"))
    sys.stderr = Tee(sys.stderr, open(console_log_path, "a"))
    # Initialize TensorBoard writer and metrics CSV
    writer = SummaryWriter(LOG_DIR)
    if not os.path.exists(EPISODE_METRICS_CSV):
        with open(EPISODE_METRICS_CSV, 'w') as f:
            f.write(
                'Episode,Reward,Avg_Reward,Length,Epsilon,Loss,TD_Error_Mean,Q_Value_Max,Q_Value_Mean,Grad_Norm,Learning_Rate,Frames_per_Sec\n'
            )
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint .pth file to resume training')
    parser.add_argument('--resume-frames', type=int, default=None, help='Manually specify total frames when resuming from an old checkpoint')
    parser.add_argument('--noisy', action='store_true', help='Enable NoisyNets for exploration (overrides epsilon-greedy)')
    args = parser.parse_args()

    # Set device, preferring MPS on Apple Silicon
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")


    # Create environment without rendering for training
    env = make_env(ENV_NAME)
    state_shape = env.observation_space.shape
    
    # Ensure we can safely access state_shape and action_space.n
    if state_shape is None:
        raise ValueError("Observation space shape is None!")
    
    # Safe access to num_actions with proper type annotation
    from gymnasium.spaces import Discrete
    if isinstance(env.action_space, Discrete):
        num_actions = env.action_space.n
    else:
        print("Warning: action_space is not a Discrete space. Using default value of 4.")
        num_actions = 4  # Default to 4 actions for Atari
    
    print(f"State shape from space: {state_shape}, Number of actions: {num_actions}")
    
    # IMPORTANT: Use the state_shape directly since it's already in the correct format (4, 84, 84)
    # No need to transpose/correct it any further
    proper_input_shape = state_shape
    print(f"Input shape for model: {proper_input_shape}")
    

    # Create Agent with the correct shape
    agent = DQNAgent(input_shape=proper_input_shape, num_actions=num_actions, device=device, use_noisy=args.noisy)

    # Set model save directory based on NoisyNets flag
    if args.noisy:
        noisy_model_dir = os.path.join(MODEL_DIR, 'noisy_models')
        os.makedirs(noisy_model_dir, exist_ok=True)
        model_save_dir = noisy_model_dir
    else:
        model_save_dir = MODEL_DIR

    # --- Resume logic ---
    resume_episode = 0
    resume_total_steps = 0
    if args.resume is not None:
        print(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = agent.load(args.resume)
        import re
        m = re.search(r'episode_(\d+)', args.resume)
        if m:
            resume_episode = int(m.group(1))
        # Always check for frames.npy if available
        import numpy as np
        import os
        frames_path = os.path.join(LOG_DIR, 'frames.npy')
        frames_count = None
        if os.path.exists(frames_path):
            try:
                arr = np.load(frames_path)
                if isinstance(arr, np.ndarray) and arr.size > 0:
                    frames_count = int(arr[-1])
            except Exception as e:
                print(f"Warning: Could not load frames.npy: {e}")
        if args.resume_frames is not None:
            resume_total_steps = int(args.resume_frames)
            print(f"Using manually specified total_steps: {resume_total_steps}")
        elif frames_count is not None:
            resume_total_steps = frames_count
            print(f"Using total_steps from frames.npy: {resume_total_steps}")
        elif checkpoint is not None and 'total_steps' in checkpoint:
            resume_total_steps = int(checkpoint['total_steps'])
        else:
            m = re.search(r'frames_(\d+)', args.resume)
            if m:
                resume_total_steps = int(m.group(1))
        print(f"Resuming from episode {resume_episode}, total_steps {resume_total_steps}")

    # Optionally, try to load scores and epsilon history if resuming
    scores = []
    scores_window = deque(maxlen=100)
    eps_history = []
    avg_scores = []
    total_steps = resume_total_steps
    i_episode = resume_episode

    start_time = time.time()

    # Create a figure for real-time plotting
    # plt.figure(figsize=(12, 5))
    # ax1 = plt.subplot(121)
    # ax2 = plt.subplot(122)
    # plt.suptitle(\'DQN Training Progress\')
    # plt.show(block=False)

    print("Starting Training...")

    # If resuming, adjust the frame budget so we only train for the remaining frames
    frames_remaining = TOTAL_FRAMES_TO_TRAIN - total_steps
    final_frame_target = total_steps + frames_remaining

    while total_steps < final_frame_target:
        i_episode += 1
        state, info = env.reset() # Gymnasium returns state, info
        score = 0
        episode_steps = 0
        episode_start = time.time()

        # For per-episode stats
        episode_losses = []
        episode_td_errors = []
        episode_q_values = []
        episode_grad_norms = []
        
        while True: # Loop until episode is done
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            # clip reward to [-1,1] for lower variance
            reward_float = float(reward)  # Convert to scalar float if it's not already
            reward_clipped = max(-1.0, min(1.0, reward_float))  # Manual clipping to avoid np.clip issues with scalars
            done = terminated or truncated # Episode ends if terminated or truncated

            # Store experience in replay buffer
            agent.step(state, action, reward_clipped, next_state, done)

            state = next_state
            score += reward_clipped
            total_steps += 1
            episode_steps += 1

            # Learn after every agent step, as per original paper (if buffer is large enough)
            if len(agent.memory) >= BATCH_SIZE:
                learn_result = agent.learn(return_stats=True)
                if learn_result is not None and isinstance(learn_result, dict):
                    if 'loss' in learn_result:
                        episode_losses.append(learn_result['loss'])
                    if 'td_error_mean' in learn_result:
                        episode_td_errors.append(learn_result['td_error_mean'])
                    if 'q_value_max' in learn_result:
                        episode_q_values.append(learn_result['q_value_max'])
                    if 'q_value_mean' in learn_result:
                        episode_q_values.append(learn_result['q_value_mean'])
                    if 'grad_norm' in learn_result:
                        episode_grad_norms.append(learn_result['grad_norm'])

            state = next_state
            score += reward_clipped
            total_steps += 1
            episode_steps += 1

            # Force end episodes after MAX_T steps to avoid infinite loops
            if done or episode_steps >= MAX_T:
                episode_time = time.time() - episode_start
                break

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps_history.append(agent.epsilon) # save epsilon value
        avg_scores.append(np.mean(scores_window)) # save average score        # Print progress with completion percentage

        # --- Compute per-episode metrics ---
        avg_reward = np.mean(scores_window)
        episode_length = episode_steps
        epsilon = agent.epsilon
        loss = np.mean(episode_losses) if episode_losses else ''
        td_error_mean = np.mean(episode_td_errors) if episode_td_errors else ''
        q_value_max = np.max(episode_q_values) if episode_q_values else ''
        q_value_mean = np.mean(episode_q_values) if episode_q_values else ''
        grad_norm = np.mean(episode_grad_norms) if episode_grad_norms else ''
        learning_rate = agent.optimizer.param_groups[0]['lr'] if hasattr(agent, 'optimizer') else ''
        frames_per_sec = episode_steps / (time.time() - episode_start) if episode_steps > 0 else ''

        # --- Log to CSV ---
        with open(EPISODE_METRICS_CSV, 'a') as f:
            f.write(f"{i_episode},{score},{avg_reward},{episode_length},{epsilon},{loss},{td_error_mean},{q_value_max},{q_value_mean},{grad_norm},{learning_rate},{frames_per_sec}\n")

        # --- Log to TensorBoard ---
        writer.add_scalar('Reward/episode', score, i_episode)
        writer.add_scalar('Reward/avg_100', avg_reward, i_episode)
        writer.add_scalar('Episode/length', episode_length, i_episode)
        writer.add_scalar('Epsilon', epsilon, i_episode)
        if loss != '':
            writer.add_scalar('Loss', float(loss), i_episode)
        if td_error_mean != '':
            writer.add_scalar('TD_Error/mean', float(td_error_mean), i_episode)
        if q_value_max != '':
            writer.add_scalar('Q_Value/max', float(q_value_max), i_episode)
        if q_value_mean != '':
            writer.add_scalar('Q_Value/mean', float(q_value_mean), i_episode)
        if grad_norm != '':
            writer.add_scalar('Grad_Norm', float(grad_norm), i_episode)
        if learning_rate != '':
            writer.add_scalar('Learning_Rate', float(learning_rate), i_episode)
        if frames_per_sec != '':
            writer.add_scalar('Frames_per_Sec', float(frames_per_sec), i_episode)


        # Print progress and save checkpoints
        progress_pct = (total_steps / TOTAL_FRAMES_TO_TRAIN) * 100
        display_progress_pct = min(progress_pct, 100.0)
        elapsed_training_time = time.time() - start_time
        elapsed_hours, elapsed_rem = divmod(elapsed_training_time, 3600)
        elapsed_minutes, elapsed_seconds = divmod(elapsed_rem, 60)
        elapsed_str = f"{int(elapsed_hours):02d}:{int(elapsed_minutes):02d}:{int(elapsed_seconds):02d}"
        print(f'Episode {i_episode}\tScore: {score:.1f} | Avg: {np.mean(scores_window):.1f} | Frames: {total_steps}/{TOTAL_FRAMES_TO_TRAIN} ({display_progress_pct:.1f}%) | Eps: {agent.epsilon:.2f} | Elapsed: {elapsed_str}')

        # Save a more frequent record of training progress
        if total_steps % 50000 == 0:  # Every 50K frames, save checkpoint and progress
            np.save(os.path.join(LOG_DIR, 'scores.npy'), np.array(scores))
            np.save(os.path.join(LOG_DIR, 'eps_history.npy'), np.array(eps_history))
            np.save(os.path.join(LOG_DIR, 'frames.npy'), np.array([total_steps]))
            model_save_path = os.path.join(model_save_dir, f"demonattack_dqn_frames_{total_steps}.pth")
            agent.save(model_save_path, total_steps=total_steps)
            print(f"Progress checkpoint saved at {total_steps} frames.")

        # Save model checkpoint
        if i_episode % SAVE_EVERY == 0:
            model_save_path = os.path.join(model_save_dir, f"demonattack_dqn_episode_{i_episode}.pth")
            agent.save(model_save_path, total_steps=total_steps)
            # --- Generate and save a plot of scores so far ---
            plt.figure()
            plt.plot(np.arange(len(scores)), scores)
            if len(scores) >= 100:
                moving_avg = np.convolve(scores, np.ones(100)/100, mode='valid')
                plt.plot(np.arange(len(moving_avg)) + 99, moving_avg, label='Moving Avg (100 episodes)')
            plt.ylabel('Score')
            plt.xlabel('Episode #')
            plt.title(f'DQN Training Scores - {ENV_NAME} (up to ep {i_episode})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(LOG_DIR, f'training_plot_episode_{i_episode}.png'))
            plt.close()

    # End of training loop: close TensorBoard writer
    writer.close()

    print("\nTraining finished.")
    env.close()

    # --- Log and print total training time ---
    total_training_time = time.time() - start_time
    hours, rem = divmod(total_training_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} (hh:mm:ss)"
    print(f"Total wall-clock training time: {time_str} ({total_training_time:.2f} seconds)")
    # Save to a text file for later analysis
    with open(os.path.join(LOG_DIR, "training_time.txt"), "w") as f:
        f.write(f"Total training time: {time_str} ({total_training_time:.2f} seconds)\n")

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

    # --- Generate and save all summary plots/statistics ---
    try:
        from utils import create_training_summary
        create_training_summary(log_file="training_log.csv", output_dir=LOG_DIR)
        print(f"Training summary and plots saved in {LOG_DIR}")
    except Exception as e:
        print(f"Could not generate training summary: {e}")