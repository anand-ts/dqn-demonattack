import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import ale_py
from matplotlib import animation
from typing import List, Any, Tuple, Union, Optional
from matplotlib.artist import Artist

# Import from our project
from dqn_agent import DQNAgent
from main import make_env, ENV_NAME

def save_frames_as_gif(frames, path='./results/movie.gif', fps=30):
    """Save a list of frames as a gif"""
    # Double the output GIF size
    plt.figure(figsize=(2 * frames[0].shape[1] / 72.0, 2 * frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    
    def animate(i) -> List[Artist]:
        patch.set_data(frames[i])
        return [patch]  # Return a list containing the artist
    
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path, writer='pillow', fps=fps)
    print(f"GIF saved to {path}")

def visualize_agent(model_path, num_episodes=3, render_mode='rgb_array'):
    """Visualize a trained agent"""
    # Create environment with rendering
    env = make_env(ENV_NAME, render_mode=render_mode)
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n  # type: ignore # Access to Discrete action space's n attribute
    
    # Create agent and load model
    device = torch.device("cpu")  # Use CPU for visualization
    agent = DQNAgent(input_shape=state_shape, num_actions=num_actions, device=device)
    agent.load(model_path)
    agent.epsilon = 0.0  # No exploration, just exploitation
    
    # Run episodes
    for i_episode in range(num_episodes):
        state, info = env.reset()
        frames = []
        score: float = 0.0  # Initialize as float to avoid type issues
        done = False
        
        while not done:
            # Capture frame if using rgb_array
            if render_mode == 'rgb_array':
                frames.append(env.render())
            else:
                env.render()  # Just render to screen
            
            # Select action (no exploration)
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            score += float(reward)  # Ensure reward is converted to float
            
            # Add small delay if rendering to screen
            if render_mode == 'human':
                time.sleep(0.01)
        
        print(f"Episode {i_episode+1} Score: {score}")
        
        # Save as gif if using rgb_array
        if render_mode == 'rgb_array' and len(frames) > 0:
            os.makedirs('./results', exist_ok=True)
            save_frames_as_gif(frames, f'./results/episode_{i_episode+1}.gif')
    
    env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize a trained DQN agent')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to visualize')
    parser.add_argument('--render', choices=['human', 'rgb_array'], default='rgb_array', 
                       help='Render mode (human for real-time, rgb_array for GIF)')
    
    args = parser.parse_args()
    visualize_agent(args.model, args.episodes, args.render)
