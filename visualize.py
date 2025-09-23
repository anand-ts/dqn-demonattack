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
import cv2  # For displaying frames with correct aspect ratio

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

def visualize_agent(model_path, num_episodes=3, render_mode='rgb_array', scale: int = 3, viewer: str = 'opencv'):
    """Visualize a trained agent"""
    # Choose rendering mode based on viewer preference
    # - viewer='opencv': use rgb_array + OpenCV window (no audio, crisp aspect ratio)
    # - viewer='native': use env's human renderer (may stretch; audio depends on ALE support)
    if render_mode == 'human' and viewer == 'native':
        env_render_mode = 'human'
    else:
        env_render_mode = 'rgb_array' if render_mode in ('human', 'rgb_array') else render_mode

    # Create environment with appropriate render mode
    env = make_env(ENV_NAME, render_mode=env_render_mode)
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
            # Capture/display frames
            if env_render_mode == 'rgb_array':
                frame = env.render()
                if frame is None:
                    # Nothing to render this step
                    pass
                else:
                    # Some envs may return lists; take the last frame if so
                    if isinstance(frame, list) and len(frame) > 0:
                        frame = frame[-1]
                    # Ensure numpy array
                    frame_np = np.asarray(frame)

                    if render_mode == 'rgb_array':
                        # Save for GIF; expect HxWxC RGB uint8
                        frames.append(frame_np)
                    elif render_mode == 'human' and viewer == 'opencv':
                        # Show in a window using OpenCV with preserved aspect ratio
                        # Normalize dtype
                        if frame_np.dtype != np.uint8:
                            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                        # Convert to BGR for OpenCV depending on channels
                        if frame_np.ndim == 2:
                            bgr = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2BGR)
                        elif frame_np.ndim == 3 and frame_np.shape[-1] == 3:
                            bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                        elif frame_np.ndim == 3 and frame_np.shape[-1] == 4:
                            bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2BGR)
                        else:
                            # Unexpected shape; try best-effort display
                            bgr = frame_np if frame_np.ndim == 2 else frame_np[..., :3]

                        if scale and scale > 1:
                            h, w = bgr.shape[:2]
                            bgr = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
                        cv2.imshow('DemonAttack (viewer)', bgr)
                        # A small wait to refresh window and allow quit with 'q'
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            done = True
                            break
            else:
                # Fallback: native viewer
                env.render()
            
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
            os.makedirs('./public', exist_ok=True)
            save_frames_as_gif(frames, f'./public/episode_{i_episode+1}.gif')
    
    env.close()
    # Ensure any OpenCV windows are closed
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize a trained DQN agent')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to visualize')
    parser.add_argument('--render', choices=['human', 'rgb_array'], default='rgb_array', 
                       help='Render mode (human shows a live window, rgb_array saves a GIF)')
    parser.add_argument('--scale', type=int, default=3, help='Scale factor for the display window (human mode)')
    parser.add_argument('--viewer', choices=['opencv', 'native'], default='opencv',
                       help='Viewer for human mode: opencv (no audio, better aspect) or native (ALE window, audio if supported)')
    
    args = parser.parse_args()
    visualize_agent(args.model, args.episodes, args.render, args.scale, args.viewer)
