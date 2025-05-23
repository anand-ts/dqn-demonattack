# utils.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd  # type: ignore
import os
import time
from typing import Optional, Union, List, Any, Tuple

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
                # Fix for "__getitem__" method not defined on type "Axes"
                ax = axes[i] if isinstance(axes, np.ndarray) else axes
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
                # Fix for "__getitem__" method not defined on type "Axes"
                ax = axes[i] if isinstance(axes, np.ndarray) else axes
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

def log_training_stats(episode, score, avg_score, epsilon, steps, elapsed_time, 
                       loss=None, lr=None, log_file="training_log.csv"):
    """Log training statistics to a CSV file with enhanced metrics"""
    import csv
    import os
    
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, 'a', newline='') as csvfile:
        headers = ['Episode', 'Score', 'Avg_Score', 'Epsilon', 'Steps', 'Time', 'Loss', 'LR']
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'Episode': episode,
            'Score': score,
            'Avg_Score': avg_score,
            'Epsilon': epsilon,
            'Steps': steps,
            'Time': elapsed_time,
            'Loss': loss if loss is not None else '',
            'LR': lr if lr is not None else ''
        })

def create_training_summary(log_file="training_log.csv", output_dir="./results"):
    """Create a summary of training progress with visualizations"""
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found")
        return
    
    # Load the training data
    df = pd.read_csv(log_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizations
    plt.style.use('ggplot')
    
    # 1. Scores over time with moving average
    plt.figure(figsize=(12, 6))
    plt.plot(df['Episode'], df['Score'], alpha=0.4, label='Score')
    
    # Calculate moving average with pandas
    window_size = min(100, len(df))
    if window_size > 0:
        df['Moving_Avg'] = df['Score'].rolling(window=window_size).mean()
        plt.plot(df['Episode'], df['Moving_Avg'], linewidth=2, label=f'Moving Avg ({window_size} ep)')
    
    plt.title('Training Scores Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'scores_over_time.png'))
    
    # 2. Create a training progress figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DQN Training Progress Summary', fontsize=16)
    
    # Plot 1: Scores with moving average
    ax = axes[0, 0]
    ax.plot(df['Episode'], df['Score'], alpha=0.4, color='lightblue')
    if 'Moving_Avg' in df.columns:
        ax.plot(df['Episode'], df['Moving_Avg'], linewidth=2, color='blue')
    ax.set_title('Training Scores')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    
    # Plot 2: Epsilon over time
    ax = axes[0, 1]
    ax.plot(df['Episode'], df['Epsilon'], color='green')
    ax.set_title('Exploration Rate (Epsilon)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    
    # Plot 3: Steps per episode
    ax = axes[1, 0]
    ax.plot(df['Episode'], df['Steps'], color='purple')
    ax.set_title('Steps per Episode')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    
    # Plot 4: Loss if available
    ax = axes[1, 1]
    if 'Loss' in df.columns and not df['Loss'].isnull().all():
        ax.plot(df['Episode'], df['Loss'], color='red')
        ax.set_title('Training Loss')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
    else:
        ax.text(0.5, 0.5, 'Loss data not available', 
               horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(output_dir, 'training_summary.png'))
    
    # Print some statistics
    print("\n=== Training Summary ===")
    print(f"Number of episodes: {len(df)}")
    print(f"Max score: {df['Score'].max()}")
    print(f"Average score (last 100 episodes): {df['Score'].tail(100).mean()}")
    print(f"Average score (all episodes): {df['Score'].mean()}")
    
    return df
