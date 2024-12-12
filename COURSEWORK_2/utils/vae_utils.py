import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
import seaborn as sns

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

    def plot_loss_distributions(self, figsize: tuple = (12, 6)):
        """Plot the distribution of loss values using violin plots."""
        plt.figure(figsize=figsize)
        
        # Prepare data for plotting
        data = []
        labels = []
        categories = []
        
        for loss_type in ['total_loss', 'reconstruction_loss_img', 
                         'reconstruction_loss_num', 'kl_loss']:
            train_losses = self.epoch_losses[loss_type]
            data.extend(train_losses)
            labels.extend(['Train'] * len(train_losses))
            categories.extend([loss_type] * len(train_losses))
            
            if len(self.val_losses[loss_type]) > 0:
                val_losses = self.val_losses[loss_type]
                data.extend(val_losses)
                labels.extend(['Validation'] * len(val_losses))
                categories.extend([loss_type] * len(val_losses))
        
        # Create violin plot
        sns.violinplot(x=categories, y=data, hue=labels)
        plt.xticks(rotation=45)
        plt.title('Distribution of Loss Components')
        plt.xlabel('Loss Component')
        plt.ylabel('Loss Value')
        plt.tight_layout()
        plt.show()