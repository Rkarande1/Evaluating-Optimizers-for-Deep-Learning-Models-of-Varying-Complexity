import torch
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
import json
import seaborn as sns 
matplotlib.use('Agg')
from torchmetrics.classification import JaccardIndex, Accuracy, ConfusionMatrix 

# Set seed for reproducibility
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

# Metric initialization
def get_metrics(num_classes, device, ignore_index=255):
    """Returns torchmetrics objects for mIoU, Accuracy, and Confusion Matrix."""
    return {
        'mIoU': JaccardIndex(task='multiclass', num_classes=num_classes, 
                             ignore_index=ignore_index).to(device),
        'pixel_acc': Accuracy(task='multiclass', num_classes=num_classes,
                              ignore_index=ignore_index, average='micro').to(device),
        'conf_mat': ConfusionMatrix(task='multiclass', num_classes=num_classes, 
                                     normalize='true', ignore_index=ignore_index).to(device)
    }

# Early stopping logic
class EarlyStopping:
    """Manages early stopping during training."""
    def __init__(self, patience=10, min_delta=0.001, mode='min', verbose=True, warmup=5):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.warmup = warmup
        self.counter = 0
        self.best_score = None 
        self.early_stop = False

        if self.mode == 'min':
            self.best_score = float('inf')
        elif self.mode == 'max':
            self.best_score = float('-inf')
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def __call__(self, current_score, epoch):
        if epoch < self.warmup:
            if self.verbose: print(f"EarlyStopping: Warmup (Epoch {epoch+1}/{self.warmup}).")
            return False

        if np.isnan(current_score):
            if self.verbose: print("EarlyStopping: Metric is NaN. Counting.")
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
            return self.early_stop

        if self.best_score is None or \
           (self.mode == 'min' and current_score < self.best_score - self.min_delta) or \
           (self.mode == 'max' and current_score > self.best_score + self.min_delta):
            if self.verbose and self.best_score is not None:
                print(f"EarlyStopping: Metric improved from {self.best_score:.4f} to {current_score:.4f}. Reset.")
            self.best_score = current_score
            self.counter = 0
        else:
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter + 1}/{self.patience} epochs (best: {self.best_score:.4f}).")
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        return self.early_stop

    def state_dict(self):
        """Returns EarlyStopping's current state."""
        return {k: getattr(self, k) for k in ["patience", "min_delta", "mode", "verbose", "warmup", "counter", "best_score", "early_stop"]}

    def load_state_dict(self, state_dict):
        """Loads EarlyStopping's state."""
        for k, v in state_dict.items(): setattr(self, k, v)


# Save / Load Checkpoints
def save_checkpoint(state, filename="checkpoint.pth"):
    """Saves model checkpoint."""
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    try:
        torch.save(state, filename)
        print(f"Checkpoint saved to {filename}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}. Trying fallback.")
        torch.save(state, os.path.join(os.path.dirname(filename) or '.', "fallback_checkpoint.pth"))
        print(f"Fallback checkpoint saved.")


def load_checkpoint(filepath, model, optimizer=None, scaler=None, early_stopper=None, device='cpu'):
    """Loads model checkpoint."""
    print(f"Loading checkpoint from {filepath}")
    if not os.path.exists(filepath):
        print(f"Checkpoint not found at: {filepath}")
        return None
    try:
        checkpoint = torch.load(filepath, map_location=device)
        if model and "model_state_dict" in checkpoint: model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint: optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler and "scaler_state_dict" in checkpoint: scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if early_stopper and "early_stopper_state_dict" in checkpoint: early_stopper.load_state_dict(checkpoint["early_stopper_state_dict"])
        print("Checkpoint loaded.")
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

# Plotting Function
def plot_losses(train_losses, val_losses, val_mious, val_pixel_accuracies,
                save_path="training_metrics.png", optimizer_name=None, epoch_offset=0):
    """Plots training/validation metrics."""
    epochs = range(1 + epoch_offset, len(train_losses) + 1 + epoch_offset)
    
    plt.figure(figsize=(15, 5)) # Adjusted for 3 plots
    plt.suptitle(f"Training Metrics ({optimizer_name or 'Default'})", fontsize=16)

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_mious, label='Validation mIoU', color='orange')
    plt.title('Validation Mean IoU')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_pixel_accuracies, label='Validation Pixel Accuracy', color='purple')
    plt.title('Validation Pixel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect for suptitle
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training metrics plot saved to {save_path}")

# Plotting function for combined optimizer metrics
def plot_combined_optimizer_metrics(trials_data, save_path="combined_optimizer_metrics.png"):
    """Plots validation mIoU and Pixel Accuracy for multiple optimizers."""
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for data in trials_data:
        epochs = range(1, len(data['val_mious']) + 1)
        plt.plot(epochs, data['val_mious'], label=f"{data['optimizer_name']} mIoU")
    plt.title('Validation Mean IoU Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for data in trials_data:
        epochs = range(1, len(data['val_pixel_accuracies']) + 1)
        plt.plot(epochs, data['val_pixel_accuracies'], label=f"{data['optimizer_name']} Pixel Acc")
    plt.title('Validation Pixel Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Accuracy')
    plt.legend()
    plt.grid(True)

    plt.suptitle("Optimizer Performance Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect for suptitle

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Combined optimizer metrics plot saved to {save_path}")

# Optimizer Configuration Tracking
def save_optimizer_config(optimizer, config_filename="optimizer_config.json"):
    """Saves optimizer config."""
    config = {'type': type(optimizer).__name__, 'params': optimizer.defaults}
    os.makedirs(os.path.dirname(config_filename) or '.', exist_ok=True)
    with open(config_filename, 'w') as f: json.dump(config, f, indent=4)
    print(f"Optimizer config saved to {config_filename}")

# Memory Profiling
def log_gpu_memory():
    """Prints GPU memory usage."""
    if torch.cuda.is_available():
        print(f"GPU Memory - Allocated: {torch.cuda.memory_allocated()/1e6:.2f}MB, "
              f"Cached: {torch.cuda.memory_reserved()/1e6:.2f}MB")
    else:
        print("GPU not available.")

# Segmentation Visualization
PASCAL_VOC_COLORMAP = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128] # Class 20
]

PASCAL_VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "void/border"
]

def decode_seg_map_sequence(mask_batch):
    """Decodes class IDs to RGB images using Pascal VOC colormap."""
    rgb_masks = []
    for mask in mask_batch:
        rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for class_id, color in enumerate(PASCAL_VOC_COLORMAP):
            rgb_mask[mask == class_id] = color
        rgb_mask[mask == 255] = [0, 0, 0] # Void/ignore class is black
        rgb_masks.append(rgb_mask)
    return torch.from_numpy(np.array(rgb_masks))


def visualize_predictions(model, dataloader, device, num_samples=5, save_dir=None, filename_prefix="test_prediction"):
    """Visualizes model predictions against ground truth."""
    model.eval()
    sample_count = 0
    if save_dir: os.makedirs(save_dir, exist_ok=True)

    # Denormalization constants
    mean = np.array([0.45677424265616295, 0.443102272306289, 0.4082499674586274])
    std = np.array([0.2369761136780325, 0.2332828798308419, 0.23898276282840822])

    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            predictions = model(images)
            predicted_masks = torch.argmax(predictions, dim=1)

            for j in range(images.shape[0]):
                if sample_count >= num_samples: break

                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original Image (denormalized)
                img_display = images[j].cpu().numpy().transpose((1, 2, 0))
                img_display = std * img_display + mean
                img_display = np.clip(img_display, 0, 1)

                axs[0].imshow(img_display)
                axs[0].set_title("Original Image")
                axs[0].axis('off')

                # Ground Truth Mask
                axs[1].imshow(decode_seg_map_sequence(masks[j].unsqueeze(0)).squeeze(0).numpy())
                axs[1].set_title("Ground Truth Mask")
                axs[1].axis('off')

                # Predicted Mask
                axs[2].imshow(decode_seg_map_sequence(predicted_masks[j].unsqueeze(0)).squeeze(0).numpy())
                axs[2].set_title("Predicted Mask")
                axs[2].axis('off')

                plt.tight_layout()
                save_path_full = os.path.join(save_dir, f"{filename_prefix}_{sample_count:02d}.png") if save_dir else None
                if save_path_full:
                    plt.savefig(save_path_full)
                    plt.close(fig)
                else:
                    plt.show()

                sample_count += 1
            if sample_count >= num_samples: break
    print(f"\nFinished visualizing {sample_count} predictions.")


def plot_confusion_matrix(conf_mat_tensor, num_classes, class_labels, save_path="confusion_matrix.png"):
    """Plots normalized confusion matrix heatmap."""
    plt.figure(figsize=(num_classes + 2, num_classes + 2))
    
    conf_mat_np = conf_mat_tensor.cpu().numpy()
    sns.heatmap(conf_mat_np, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_labels[:num_classes], 
                yticklabels=class_labels[:num_classes],
                cbar_kws={'label': 'Normalized Frequency'})
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to {save_path}")