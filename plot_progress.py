import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
import re
import pandas as pd

def extract_numbers(filename):
    """Extract episode number from filename using regex"""
    match = re.search(r'episode_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

def plot_training_progress(log_dir='./results', csv_file=None):
    """Plot training progress from saved files"""
    plt.style.use('ggplot')  # Use a nicer style
    
    if csv_file and os.path.exists(csv_file):
        # Load from CSV if available (better format with multiple metrics)
        print(f"Loading training data from CSV: {csv_file}")
        df = pd.read_csv(csv_file)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DQN Training Progress', fontsize=16)
        
        # Plot 1: Scores
        ax = axes[0, 0]
        ax.plot(df['Episode'], df['Score'], alpha=0.4, color='lightblue', label='Raw Score')
        # Calculate and plot moving average
        window_size = min(100, len(df))
        if window_size > 0:
            df['Moving_Avg'] = df['Score'].rolling(window=window_size).mean()
            ax.plot(df['Episode'], df['Moving_Avg'], linewidth=2, color='blue', label=f'Moving Avg ({window_size} ep)')
        ax.set_title('Training Scores')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.legend()
        
        # Plot 2: Epsilon
        ax = axes[0, 1]
        ax.plot(df['Episode'], df['Epsilon'], color='green')
        ax.set_title('Exploration Rate (Epsilon)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        
        # Plot 3: Time per episode
        ax = axes[1, 0]
        if 'Time' in df.columns:
            ax.plot(df['Episode'], df['Time'], color='orange')
            ax.set_title('Time per Episode')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Seconds')
        else:
            ax.text(0.5, 0.5, 'Time data not available', 
                   horizontalalignment='center', verticalalignment='center')
        
        # Plot 4: Steps per episode
        ax = axes[1, 1]
        if 'Steps' in df.columns:
            ax.plot(df['Episode'], df['Steps'], color='purple')
            ax.set_title('Steps per Episode')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Steps')
        else:
            ax.text(0.5, 0.5, 'Step data not available', 
                   horizontalalignment='center', verticalalignment='center')
        
    else:
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
        
        # Get epsilon history if available
        eps_file = os.path.join(log_dir, 'eps_history.npy')
        has_eps = os.path.exists(eps_file)
        
        # Create figure with additional subplot if epsilon data exists
        if has_eps:
            fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        else:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
        plt.suptitle('DQN Training Progress', fontsize=16)
        
        # Plot 1: Raw scores
        ax = axes[0]
        ax.plot(x, scores, alpha=0.4, color='lightblue', label='Raw Score')
        ax.set_title('Raw Training Scores')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        
        # Plot 2: Moving average with proper visualization
        ax = axes[1]
        window_size = min(100, len(scores))
        if window_size > 0:
            moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
            # Plot the moving average
            ax.plot(np.arange(len(moving_avg)) + window_size-1, moving_avg, 
                   linewidth=2, color='blue', label=f'Moving Average (Window Size: {window_size})')
            # Add a horizontal line showing the final average
            if len(moving_avg) > 0:
                ax.axhline(y=moving_avg[-1], color='red', linestyle='--', 
                          label=f'Final Avg: {moving_avg[-1]:.1f}')
        ax.set_title(f'Moving Average (Window Size: {window_size})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Score')
        ax.legend()
        
        # Plot 3: Epsilon history if available
        if has_eps:
            eps_history = np.load(eps_file)
            ax = axes[2]
            ax.plot(np.arange(len(eps_history)), eps_history, color='green')
            ax.set_title('Exploration Rate (Epsilon)')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Epsilon')
    
    plt.tight_layout()
    # Adjust spacing between subplots
    plt.subplots_adjust(top=0.92, hspace=0.3)
    
    # Save the plot
    output_path = os.path.join(log_dir, 'training_progress_detailed.png')
    plt.savefig(output_path)
    plt.show()
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot DQN training progress')
    parser.add_argument('--dir', type=str, default='./results', help='Directory containing result files')
    parser.add_argument('--csv', type=str, default=None, help='CSV file with training data')
    
    args = parser.parse_args()
    plot_training_progress(args.dir, args.csv)
