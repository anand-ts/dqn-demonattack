# utils.py
import numpy as np
import torch
import matplotlib.pyplot as plt

def debug_observation(observation, title="Observation", save_path=None):
    """
    Debug an observation from the environment
    
    Args:
        observation: The observation to debug (numpy array)
        title: Title for the plot
        save_path: Path to save the visualization (if None, just displays)
    """
    print(f"{title} shape: {observation.shape}, dtype: {observation.dtype}, min: {np.min(observation)}, max: {np.max(observation)}")
    
    # If it's a stacked frame observation with channels first
    if len(observation.shape) == 3 and observation.shape[0] <= 4:
        fig, axes = plt.subplots(1, observation.shape[0], figsize=(4*observation.shape[0], 4))
        for i in range(observation.shape[0]):
            if observation.shape[0] == 1:
                ax = axes
            else:
                ax = axes[i]
            ax.imshow(observation[i], cmap='gray')
            ax.set_title(f"Channel {i}")
            ax.axis('off')
    
    # If channels last
    elif len(observation.shape) == 3 and observation.shape[2] <= 4:
        fig, axes = plt.subplots(1, observation.shape[2], figsize=(4*observation.shape[2], 4))
        for i in range(observation.shape[2]):
            if observation.shape[2] == 1:
                ax = axes
            else:
                ax = axes[i]
            ax.imshow(observation[:,:,i], cmap='gray')
            ax.set_title(f"Channel {i}")
            ax.axis('off')
    
    else:
        # For other shapes, just show the first channel/frame
        plt.figure(figsize=(6, 6))
        plt.imshow(observation.reshape(observation.shape[0], -1), cmap='gray')
        plt.title(f"Reshaped observation")
        plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
