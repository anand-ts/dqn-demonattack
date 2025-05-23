import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
import seaborn as sns  # type: ignore
import torch
from typing import Optional, Dict, Any, List, Union

def load_training_data(csv_file="training_log.csv"):
    """Load training data from CSV file"""
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} not found")
        return None
    
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def create_comprehensive_visualization(data, output_dir="./results", model_info=None):
    """Create comprehensive visualizations of training data"""
    if data is None or len(data) == 0:
        print("No data to visualize")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('ggplot')
    sns.set_context("notebook", font_scale=1.2)
    
    # 1. Create a dashboard figure with multiple plots
    fig = plt.figure(figsize=(20, 15))
    title = 'DQN Training Analysis Dashboard'
    if model_info and 'use_double_dqn' in model_info and model_info['use_double_dqn']:
        title = 'Double DQN Training Analysis Dashboard'
    fig.suptitle(title, fontsize=20, y=0.98)
    
    # Define grid for subplots
    gs = fig.add_gridspec(3, 3)
    
    # Plot 1: Score progression with moving average (larger plot)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(data['Episode'], data['Score'], alpha=0.4, color='lightblue', label='Raw Score')
    
    # Calculate moving average with pandas
    window_sizes = [10, 50, 100]
    for window in window_sizes:
        if len(data) >= window:
            data[f'MA_{window}'] = data['Score'].rolling(window=window).mean()
            ax1.plot(data['Episode'], data[f'MA_{window}'], 
                    linewidth=2, label=f'Moving Avg ({window} ep)')
    
    ax1.set_title('Score Progression', fontsize=16)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Epsilon decay
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(data['Episode'], data['Epsilon'], color='green')
    ax2.set_title('Exploration Rate (Epsilon)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Steps per episode
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(data['Episode'], data['Steps'], color='purple')
    ax3.set_title('Steps per Episode')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: Time per episode
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(data['Episode'], data['Time'], color='orange')
    ax4.set_title('Time per Episode')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Time (seconds)')
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 5: Loss if available
    ax5 = fig.add_subplot(gs[2, 0])
    if 'Loss' in data.columns and not data['Loss'].isnull().all():
        # Filter out NaN values
        loss_data = data.dropna(subset=['Loss'])
        if len(loss_data) > 0:
            ax5.plot(loss_data['Episode'], loss_data['Loss'], color='red')
            ax5.set_title('Training Loss')
            ax5.set_xlabel('Episode')
            ax5.set_ylabel('Loss')
            # Use log scale if loss values vary greatly
            if loss_data['Loss'].max() / (loss_data['Loss'].min() + 1e-10) > 100:
                ax5.set_yscale('log')
        else:
            ax5.text(0.5, 0.5, 'Loss data contains only NaN values', 
                   horizontalalignment='center', verticalalignment='center')
    else:
        ax5.text(0.5, 0.5, 'Loss data not available', 
               horizontalalignment='center', verticalalignment='center')
    ax5.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 6: Learning rate if available
    ax6 = fig.add_subplot(gs[2, 1])
    if 'LR' in data.columns and not data['LR'].isnull().all():
        lr_data = data.dropna(subset=['LR'])
        if len(lr_data) > 0:
            ax6.plot(lr_data['Episode'], lr_data['LR'], color='blue')
            ax6.set_title('Learning Rate')
            ax6.set_xlabel('Episode')
            ax6.set_ylabel('LR')
            # Use log scale for LR
            ax6.set_yscale('log')
        else:
            ax6.text(0.5, 0.5, 'LR data contains only NaN values', 
                   horizontalalignment='center', verticalalignment='center')
    else:
        ax6.text(0.5, 0.5, 'LR data not available', 
               horizontalalignment='center', verticalalignment='center')
    ax6.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 7: Score distribution (histogram)
    ax7 = fig.add_subplot(gs[2, 2])
    sns.histplot(data=data, x='Score', kde=True, ax=ax7, color='skyblue')
    ax7.set_title('Score Distribution')
    ax7.set_xlabel('Score')
    ax7.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save the dashboard figure
    output_path = os.path.join(output_dir, 'training_dashboard.png')
    plt.savefig(output_path)
    print(f"Dashboard saved to {output_path}")
    
    # Create additional specialized plots
    
    # 1. Performance Trend Analysis
    plt.figure(figsize=(12, 8))
    
    # Calculate performance metrics
    if len(data) >= 100:
        # Calculate performance in different phases of training
        data['Training_Phase'] = pd.cut(data['Episode'], 
                                       bins=[0, len(data)//4, len(data)//2, 3*len(data)//4, len(data)], 
                                       labels=['Early', 'Mid-Early', 'Mid-Late', 'Late'])
        
        # Box plot of scores by training phase
        sns.boxplot(x='Training_Phase', y='Score', data=data)
        plt.title('Score Distribution by Training Phase')
        plt.savefig(os.path.join(output_dir, 'score_by_phase.png'))
    
    # Return the dashboard figure
    plt.close('all')
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize DQN Training Data')
    parser.add_argument('--csv', type=str, default='training_log.csv', help='Path to training log CSV file')
    parser.add_argument('--output', type=str, default='./results', help='Directory for output visualizations')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint for additional information')
    
    args = parser.parse_args()
    
    # Load training data
    data = load_training_data(args.csv)
    
    # Load model info if model path is provided
    model_info = None
    if args.model and os.path.exists(args.model):
        try:
            model_info = torch.load(args.model, map_location='cpu')
            print("Loaded model information for visualization context")
        except Exception as e:
            print(f"Could not load model information: {e}")
    
    # Generate visualizations
    if data is not None:
        create_comprehensive_visualization(data, args.output, model_info)
        print(f"Visualizations generated in {args.output}")
    
if __name__ == "__main__":
    main()
