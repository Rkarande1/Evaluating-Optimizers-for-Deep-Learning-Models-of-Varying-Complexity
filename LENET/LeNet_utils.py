import torch
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
import json
import seaborn as sns
from torchvision import datasets, transforms 
from torchmetrics import Accuracy, ConfusionMatrix
from torch.utils.data import DataLoader, random_split


matplotlib.use('Agg')

# - CIFAR-10 Class Names (for plotting) -
CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# -EarlyStopping Class -
class EarlyStopping:
    """Early stops training if metric doesn't improve after patience."""
    def __init__(self, patience=10, min_delta=0.001, mode='min', verbose=True, warmup=5):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.warmup = warmup
        
        self.counter = 0
        self.early_stop = False
        self.best_score = None 
        self._current_epoch = -1 

        if self.mode == 'min':
            self.val_score_sign = 1 
            self.best_score = float('inf')
        elif self.mode == 'max':
            self.val_score_sign = -1
            self.best_score = float('-inf')
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def __call__(self, current_score, epoch):
        self._current_epoch = epoch 

        if self._current_epoch < self.warmup:
            if self.verbose:
                print(f"EarlyStopping: Warmup period (Epoch {self._current_epoch+1}/{self.warmup}). Disabled.")
            return False

        if np.isnan(current_score): # Handle NaN scores
            if self.verbose:
                print("EarlyStopping: metric is NaN. So Incrementing counter.")
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop

        # Apply sign for consistent comparison (minimizing signed_score)
        signed_current_score = self.val_score_sign * current_score

        if signed_current_score < self.best_score - self.min_delta:
            if self.verbose and self.best_score != float('inf') and self.best_score != float('-inf'):
                print(f"EarlyStopping: Metric improved from {self.best_score/self.val_score_sign:.4f} to {current_score:.4f}. Reset counter.")
            elif self.verbose: 
                 print(f"EarlyStopping: Metric initialized to {current_score:.4f}.")
            self.best_score = signed_current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs (best: {self.best_score/self.val_score_sign:.4f}).")
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
    
    def state_dict(self):
        """Returns EarlyStopping state."""
        return {
            'patience': self.patience,
            'min_delta': self.min_delta,
            'mode': self.mode,
            'verbose': self.verbose,
            'warmup': self.warmup,
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            '_current_epoch': self._current_epoch 
        }

    def load_state_dict(self, state_dict):
        """Loads EarlyStopping state."""
        self.patience = state_dict['patience']
        self.min_delta = state_dict['min_delta']
        self.mode = state_dict['mode']
        self.verbose = state_dict['verbose']
        self.warmup = state_dict['warmup']
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.early_stop = state_dict['early_stop']
        self._current_epoch = state_dict['_current_epoch'] 
        
        if self.mode == 'min':
            self.val_score_sign = 1
        elif self.mode == 'max':
            self.val_score_sign = -1


# --- General Utilities ---
def set_seed(seed):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

# --- Checkpoint Handling ---
def save_checkpoint(state, filename="checkpoint.pth"):
    """Saves training checkpoint."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filepath, model, optimizer=None, scaler=None, early_stopper=None, device='cpu'):
    """Loads model and optimizer state from a checkpoint."""
    if not os.path.exists(filepath):
        print(f"Checkpoint file not found at: {filepath}")
        return None
    
    print(f"Loading checkpoint from {filepath}")
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if early_stopper and "early_stopper_state_dict" in checkpoint:
        early_stopper.load_state_dict(checkpoint["early_stopper_state_dict"])
    
    print("Model, Optimizer, Scaler, and EarlyStopper states loaded (if available in checkpoint).")
    return checkpoint

def save_optimizer_config(optimizer, config_path):
    """Saves the optimizer's initial configuration."""
    config = {
        "name": type(optimizer).__name__,
        "params": optimizer.defaults
    }
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Optimizer config saved to {config_path}")


# --- Metric Handling (for Classification) ---
def get_metrics(num_classes, device):
    """Returns torchmetrics instances for classification."""
    metrics = {
        'accuracy': Accuracy(task="multiclass", num_classes=num_classes).to(device),
        'conf_mat': ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    }
    return metrics

# --- CIFAR-10 Data Loading with Custom Train/Val Split ---
def get_cifar10_dataloaders(batch_size, train_val_split_ratio=0.9, num_workers=8, data_root='data'):
    """Loads CIFAR-10 data and creates train, val, test DataLoaders."""
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    full_train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_transform)

    train_size = int(train_val_split_ratio * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    print(f"CIFAR-10 Data Split: Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)} samples")

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

# --- Plotting Functions (for Classification Metrics) ---
def plot_single_run_metrics(train_losses, val_losses, val_accuracies, 
                            confusion_matrix_tensor, save_dir, optimizer_name, epoch_offset=0,
                            num_classes=10):
    """Plots training/validation loss and accuracy, and a confusion matrix for a single run."""
    epochs = range(epoch_offset + 1, epoch_offset + len(train_losses) + 1)

    if not epochs:
        print(f"No data to plot for {optimizer_name}'s single run metrics.")
        return

    # Plot Loss and Accuracy
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
    fig.suptitle(f'{optimizer_name} - Training & Validation Metrics', fontsize=16)

    # Plot Losses
    axes[0].plot(epochs, train_losses, label='Train Loss', marker='o', linestyle='-')
    axes[0].plot(epochs, val_losses, label='Validation Loss', marker='x', linestyle='--')
    axes[0].set_title('Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot Accuracies
    axes[1].plot(epochs, val_accuracies, label='Validation Accuracy', marker='o', linestyle='-')
    axes[1].set_title('Validation Accuracy over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    metrics_plot_path = os.path.join(save_dir, f"{optimizer_name}_training_metrics.png")
    plt.savefig(metrics_plot_path, dpi=300)
    plt.close(fig)
    print(f"Training/Validation metrics plot saved to {metrics_plot_path}")

    # Plot Confusion Matrix
    if confusion_matrix_tensor is not None:
        # Use CIFAR10_CLASSES for labels
        class_labels = CIFAR10_CLASSES[:num_classes] 

        cm_numpy = confusion_matrix_tensor.cpu().numpy()
        # Normalize the confusion matrix over the true labels (rows)
        cm_normalized = cm_numpy.astype('float') / (cm_numpy.sum(axis=1)[:, np.newaxis] + 1e-6)

        fig_cm, ax_cm = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", ax=ax_cm,
                    xticklabels=class_labels, yticklabels=class_labels,
                    cbar_kws={'label': 'Normalized Frequency'})
        
        ax_cm.set_title(f'{optimizer_name} - Normalized Confusion Matrix (Test Set)')
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_ylabel('True Label')
        
        plt.tight_layout()
        cm_plot_path = os.path.join(save_dir, f"{optimizer_name}_confusion_matrix.png")
        plt.savefig(cm_plot_path, dpi=300)
        plt.close(fig_cm)
        print(f"Confusion matrix plot saved to {cm_plot_path}")
    else:
        print("Confusion matrix tensor is None.")


def plot_combined_optimizer_metrics(all_opt_data, plot_save_dir, epoch_offset=0):
    
    if not all_opt_data:
        print("No optimizer data loaded for combined plotting.")
        return

    os.makedirs(plot_save_dir, exist_ok=True)

    max_epochs = 0
    for opt_name, data in all_opt_data.items():
        max_epochs = max(max_epochs, len(data.get('val_losses', [])))

    if max_epochs == 0:
        print("No valid data to plot combined optimizer metrics.")
        return

    # Plot Validation Loss
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    for opt_name, data in all_opt_data.items():
        epochs = range(epoch_offset + 1, epoch_offset + len(data.get('val_losses', [])) + 1)
        ax_loss.plot(epochs, data.get('val_losses', []), label=f'{opt_name}', marker='o', markersize=4, linestyle='-')
    
    ax_loss.set_title('Validation Loss Comparison')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Validation Loss')
    ax_loss.legend()
    ax_loss.grid(True)
    ax_loss.set_xlim(1, max_epochs)
    plt.tight_layout()
    loss_plot_path = os.path.join(plot_save_dir, "combined_val_loss.png")
    plt.savefig(loss_plot_path, dpi=300)
    plt.close(fig_loss)
    print(f"Combined validation loss plot saved to {loss_plot_path}")

    # Plot Validation Accuracy
    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
    for opt_name, data in all_opt_data.items():
        epochs = range(epoch_offset + 1, epoch_offset + len(data.get('val_accuracies', [])) + 1)
        ax_acc.plot(epochs, data.get('val_accuracies', []), label=f'{opt_name}', marker='o', markersize=4, linestyle='--')
    
    ax_acc.set_title('Validation Accuracy Comparison')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Validation Accuracy')
    ax_acc.legend()
    ax_acc.grid(True)
    ax_acc.set_xlim(1, max_epochs)
    plt.tight_layout()
    acc_plot_path = os.path.join(plot_save_dir, "combined_val_accuracy.png")
    plt.savefig(acc_plot_path, dpi=300)
    plt.close(fig_acc)
    print(f"Combined validation accuracy plot saved to {acc_plot_path}")

