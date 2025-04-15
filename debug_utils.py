# debug_utils.py
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def print_tensor_info(name, tensor):
    """Print detailed information about a tensor or numpy array"""
    if isinstance(tensor, torch.Tensor):
        print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
        print(f"  min={tensor.min().item()}, max={tensor.max().item()}, mean={tensor.float().mean().item()}")
    elif isinstance(tensor, np.ndarray):
        print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}")
        print(f"  min={tensor.min()}, max={tensor.max()}, mean={tensor.mean()}")
    else:
        print(f"{name}: type={type(tensor)}")

def visualize_state(state, filename=None):
    """Visualize a state for debugging purposes"""
    # Create a directory for debug visualizations
    os.makedirs("debug_viz", exist_ok=True)
    
    if isinstance(state, torch.Tensor):
        state = state.cpu().numpy()
    
    # Handle different state shapes
    if len(state.shape) == 4:  # Batch of states
        state = state[0]  # Take the first state in the batch
    
    # Convert (C,H,W) to (H,W,C) for display if needed
    if state.shape[0] == 4:  # If channels first
        state_display = np.transpose(state, (1, 2, 0))
    else:
        state_display = state
    
    # Create a figure to display the frames
    plt.figure(figsize=(12, 3))
    for i in range(min(4, state_display.shape[-1])):  # Show up to 4 channels
        plt.subplot(1, 4, i+1)
        plt.imshow(state_display[..., i], cmap='gray')
        plt.title(f"Channel {i}")
        plt.axis('off')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(os.path.join("debug_viz", filename))
        plt.close()
    else:
        plt.show()
