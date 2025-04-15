import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
import re

def extract_numbers(filename):
    """Extract episode number from filename using regex"""
    match = re.search(r'episode_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

def plot_training_progress(log_dir='./results'):
    """Plot training progress from saved files"""
    # Find all score files
    score_files = glob.glob(os.path.join(log_dir, '*scores*.npy'))
    
    if not score_files:
        print(f"No score files found in {log_dir}")
        return
    
    # Sort by episode number
    score_files.sort(key=extract_numbers)
    
    # Get the latest file
    latest_file = score_files[-1]
    print(f"Loading scores from {latest_file}")
    
    scores = np.load(latest_file)
    x = np.arange(len(scores))
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Plot raw scores
    plt.subplot(211)
    plt.plot(x, scores)
    plt.title('Raw Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    # Plot moving average
    window_size = min(100, len(scores))
    if window_size > 0:
        plt.subplot(212)
        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(len(moving_avg)) + window_size-1, moving_avg)
        plt.title(f'Moving Average (Window Size: {window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Average Score')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(log_dir, 'training_progress.png')
    plt.savefig(output_path)
    plt.show()
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot DQN training progress')
    parser.add_argument('--dir', type=str, default='./results', help='Directory containing result files')
    
    args = parser.parse_args()
    plot_training_progress(args.dir)
