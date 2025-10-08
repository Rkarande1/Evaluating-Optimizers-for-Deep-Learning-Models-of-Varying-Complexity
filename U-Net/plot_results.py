
# imports
import os
import json
import numpy as np

# Third-party imports
import torch
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns 

# for non-interactive plotting
matplotlib.use('Agg')

base_output_dir = "/content/drive/MyDrive/my_project/Checkpoints/"

run_data_dir = os.path.join(base_output_dir, "run_data")

# Specify optimizer name for files
optimizer_name = "AdamW"

history_file = os.path.join(run_data_dir, f"{optimizer_name}_training_history.json")
confusion_matrix_file = os.path.join(run_data_dir, f"{optimizer_name}_confusion_matrix.pt")

# Output directory for plots
plot_output_dir = os.path.join(base_output_dir, "plots")
os.makedirs(plot_output_dir, exist_ok=True)


# --- 2. Load Saved Data ---
print(f"Loading training history from: {history_file}")
try:
    with open(history_file, 'r') as f:
        history_data = json.load(f)
    train_losses = history_data["train_losses"]
    val_losses = history_data["val_losses"]
    val_mious = history_data["val_mious"]
    val_pixel_accuracies = history_data["val_pixel_accuracies"]
    print("Training history loaded successfully.")
except FileNotFoundError:
    print(f"Error: Training history file not found at {history_file}. Check path.")
    train_losses, val_losses, val_mious, val_pixel_accuracies = [], [], [], []
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {history_file}. File might be corrupted.")
    train_losses, val_losses, val_mious, val_pixel_accuracies = [], [], [], []

print(f"Loading confusion matrix from: {confusion_matrix_file}")
try:
    final_confusion_matrix = torch.load(confusion_matrix_file)
    print("Confusion matrix loaded successfully.")
except FileNotFoundError:
    print(f"Error: Confusion matrix file not found at {confusion_matrix_file}. Check path.")
    final_confusion_matrix = None
except Exception as e:
    print(f"Error loading confusion matrix: {e}")
    final_confusion_matrix = None


# --- 3. Plotting Functions ---

def plot_losses(train_losses, val_losses, val_mious, val_pixel_accuracies, save_path, optimizer_name, epoch_offset=0):
    """Plots training and validation metrics."""
    epochs = range(epoch_offset + 1, epoch_offset + len(train_losses) + 1)

    if not epochs:
        print(f"No data to plot for {optimizer_name}.")
        return

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
    fig.suptitle(f'Training and Validation Metrics ({optimizer_name})', fontsize=16)

    # Plot Losses
    axes[0].plot(epochs, train_losses, label='Train Loss', marker='o', linestyle='-')
    axes[0].plot(epochs, val_losses, label='Validation Loss', marker='x', linestyle='--')
    axes[0].set_title('Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot mIoU and Pixel Accuracy
    axes[1].plot(epochs, val_mious, label='Validation mIoU', marker='o', linestyle='-')
    axes[1].plot(epochs, val_pixel_accuracies, label='Validation Pixel Accuracy', marker='x', linestyle='--')
    axes[1].set_title('Validation Metrics over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Metric Value')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) 
    plt.savefig(save_path, dpi=300)
    plt.close(fig) 
    print(f"Training metrics plot saved to {save_path}")


def plot_confusion_matrix(cm_tensor, num_classes, class_labels, save_path):
    """Plots a normalized confusion matrix heatmap."""
    if cm_tensor is None:
        print("Confusion matrix tensor is None. Skipping plot.")
        return

    # convert to numpy
    cm_numpy = cm_tensor.cpu().numpy()

    # Normalize confusion matrix
    cm_normalized = cm_numpy.astype('float') / (cm_numpy.sum(axis=1)[:, np.newaxis] + 1e-6)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Normalized Frequency'})

    ax.set_title('Normalized Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Confusion matrix plot saved to {save_path}")


# --- 4. Execute Plotting ---
if __name__ == "__main__":
    # Define Pascal VOC class labels
    pascal_voc_class_labels = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
        "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]

    # Plot training metrics
    if train_losses and val_losses and val_mious and val_pixel_accuracies:
        training_plot_path = os.path.join(plot_output_dir, f"{optimizer_name}_training_metrics.png")
        plot_losses(train_losses, val_losses, val_mious, val_pixel_accuracies,
                    save_path=training_plot_path,
                    optimizer_name=optimizer_name,
                    epoch_offset=0)
    else:
        print("Not enough data to plot training metrics.")

    # Plot confusion matrix
    if final_confusion_matrix is not None:
        confusion_matrix_plot_path = os.path.join(plot_output_dir, f"{optimizer_name}_confusion_matrix.png")
        plot_confusion_matrix(final_confusion_matrix,
                              num_classes=len(pascal_voc_class_labels),
                              class_labels=pascal_voc_class_labels,
                              save_path=confusion_matrix_plot_path)
    else:
        print("Confusion matrix data not available.")

    print("\nPlotting script finished.")