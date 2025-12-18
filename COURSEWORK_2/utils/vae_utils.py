import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
import seaborn as sns
import torch

class VAELossPlotter:
    def __init__(self):
        # Initialize storage for different loss components
        self.epoch_losses: Dict[str, List[float]] = {
            'total_loss': [],
            'reconstruction_loss_img': [],
            'reconstruction_loss_num': [], 
            'kl_loss': []
        }
        self.val_losses: Dict[str, List[float]] = {
            'total_loss': [],
            'reconstruction_loss_img': [],
            'reconstruction_loss_num': [],
            'kl_loss': []
        }
        self.epochs: List[int] = []
        
    def add_epoch_losses(self, 
                        epoch: int,
                        train_total_loss: float,
                        train_recon_img_loss: float,
                        train_recon_num_loss: float,
                        train_kl_loss: float,
                        val_total_loss: Optional[float] = None,
                        val_recon_img_loss: Optional[float] = None,
                        val_recon_num_loss: Optional[float] = None,
                        val_kl_loss: Optional[float] = None):
        """Add losses for a single epoch."""
        self.epochs.append(epoch)
        
        # Add training losses
        self.epoch_losses['total_loss'].append(train_total_loss)
        self.epoch_losses['reconstruction_loss_img'].append(train_recon_img_loss)
        self.epoch_losses['reconstruction_loss_num'].append(train_recon_num_loss)
        self.epoch_losses['kl_loss'].append(train_kl_loss)
        
        # Add validation losses if provided
        if val_total_loss is not None:
            self.val_losses['total_loss'].append(val_total_loss)
            self.val_losses['reconstruction_loss_img'].append(val_recon_img_loss)
            self.val_losses['reconstruction_loss_num'].append(val_recon_num_loss)
            self.val_losses['kl_loss'].append(val_kl_loss)
            
    def plot_losses(self, figsize: tuple = (15, 10), include_validation: bool = True):
        """Plot all loss components."""
        # plt.style.use('seaborn')
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('VAE Loss Components Over Training', fontsize=16)
        
        # Plot settings
        loss_names = ['Total Loss', 'Image Reconstruction Loss', 
                     'Numerical Reconstruction Loss', 'KL Divergence Loss']
        loss_keys = ['total_loss', 'reconstruction_loss_img', 
                    'reconstruction_loss_num', 'kl_loss']
        
        for idx, (name, key) in enumerate(zip(loss_names, loss_keys)):
            ax = axes[idx // 2, idx % 2]
            
            # Plot training loss
            ax.plot(self.epochs, self.epoch_losses[key], 
                   label='Training', color='blue', linewidth=2)
            
            # Plot validation loss if available and requested
            if include_validation and len(self.val_losses[key]) > 0:
                ax.plot(self.epochs, self.val_losses[key], 
                       label='Validation', color='red', linewidth=2)
            
            ax.set_title(name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()
        
    def plot_loss_ratios(self, figsize: tuple = (10, 6)):
        """Plot the ratio of each loss component to total loss."""
        plt.figure(figsize=figsize)
        
        # Calculate ratios
        total_losses = np.array(self.epoch_losses['total_loss'])
        ratios = {
            'Image Reconstruction': np.array(self.epoch_losses['reconstruction_loss_img']) / total_losses,
            'Numerical Reconstruction': np.array(self.epoch_losses['reconstruction_loss_num']) / total_losses,
            'KL Divergence': np.array(self.epoch_losses['kl_loss']) / total_losses
        }
        
        # Create stacked area plot
        plt.stackplot(self.epochs, ratios.values(),
                     labels=ratios.keys(),
                     alpha=0.8)
        
        plt.title('Loss Component Ratios Over Training')
        plt.xlabel('Epoch')
        plt.ylabel('Ratio of Total Loss')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def analyze_dataloader(dataloader, num_batches=5):
    """
    Analyze the first few batches of data to check for NaNs, zeros, and value ranges
    """
    print("\nDataLoader Analysis:")
    print("-------------------")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
            
        print(f"\nBatch {batch_idx + 1}:")
        
        # Analyze images
        images = batch.input["images"]
        print("\nImages:")
        print(f"Shape: {images.shape}")
        print(f"Type: {images.dtype}")
        print(f"Range: [{images.min().item():.3f}, {images.max().item():.3f}]")
        print(f"Mean: {images.mean().item():.3f}")
        print(f"NaN count: {torch.isnan(images).sum().item()}")
        print(f"Zero count: {(images == 0).sum().item()}")
        
        # Analyze numerical observations
        obs_types = ["ee_cartesian_pos_ob", "ee_cartesian_vel_ob", "joint_pos_ob"]
        
        for obs_type in obs_types:
            if obs_type in batch.input:
                obs = batch.input[obs_type]
                print(f"\n{obs_type}:")
                print(f"Shape: {obs.shape}")
                print(f"Type: {obs.dtype}")
                print(f"Range: [{obs.min().item():.3f}, {obs.max().item():.3f}]")
                print(f"Mean: {obs.mean().item():.3f}")
                print(f"NaN count: {torch.isnan(obs).sum().item()}")
                print(f"Zero count: {(obs == 0).sum().item()}")
                
                # Show histogram of values
                if batch_idx == 0:
                    plt.figure(figsize=(10, 4))
                    plt.hist(obs.numpy().flatten(), bins=50)
                    plt.title(f'Distribution of {obs_type} values')
                    plt.xlabel('Value')
                    plt.ylabel('Count')
                    plt.show()

    print("\nDataLoader Analysis Complete")
    print("-------------------------")