import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from LeNet_model import LeNet5
from LeNet_utils import load_checkpoint, get_metrics, get_cifar10_dataloaders, set_seed # <--- CHANGE: Import get_cifar10_dataloaders and set_seed

# CIFAR-10 specific class labels for confusion matrix 
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# --- Test Function ---
def test_model(model, test_loader, criterion, device, num_classes, metrics_dict):
    """
    Evaluates the trained model on the test set.
    """
    model.eval()
    total_test_loss = 0.0
    metrics_dict['accuracy'].reset()
    metrics_dict['conf_mat'].reset()

    print("\n--- Testing Model ---")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            logits, probas = model(images)
            loss = criterion(logits, labels)
            total_test_loss += loss.item()

            predicted_labels = torch.argmax(probas, dim=1)
            metrics_dict['accuracy'].update(predicted_labels, labels)
            metrics_dict['conf_mat'].update(predicted_labels, labels)

    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = metrics_dict['accuracy'].compute().item()
    test_confusion_matrix = metrics_dict['conf_mat'].compute()

    return avg_test_loss, test_accuracy, test_confusion_matrix


# --- Main execution block for test.py ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a LeNet5 Model on CIFAR-10")
    parser.add_argument("--model_path", type=str, required=True,
                        help="/Checkpoints/lenet_adamW_checkpoint.pth")
    parser.add_argument("--data_root", type=str, default="./data", # <-data_root arg for CIFAR-10
                        help="Root directory for CIFAR-10 dataset.")
    parser.add_argument("--num_classes", type=int, default=10, # <--10 for CIFAR-10
                        help="Number of classification classes.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for testing.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loader workers.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    set_seed(42) 

    # --- Data Loaders ---

    _, _, test_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size,
        data_root=args.data_root
    )

    print(f"Test dataset size: {len(test_loader.dataset)} samples")

    # --- Model Initialization ---
    model = LeNet5(num_classes=args.num_classes, in_channels=3).to(device) 

    # --- Load Model Checkpoint ---
    checkpoint_loaded = load_checkpoint(args.model_path, model, device=device)
    if not checkpoint_loaded:
        print(f"Failed to load model checkpoint from {args.model_path}. Please ensure the path is correct and the file exists.")
        exit() # Exit if model cannot be loaded

    # --- Loss Function for calculating test loss ---
    eval_criterion = nn.CrossEntropyLoss() 

    # --- Initialize Metrics for Testing ---
  
    test_metrics = get_metrics(args.num_classes, device)
    
    # --- Run Final Test ---
    final_test_loss, final_test_accuracy, final_confusion_matrix = test_model(
        model, test_loader, eval_criterion, device, 
        num_classes=args.num_classes, metrics_dict=test_metrics
    )

    print(f"\n--- FINAL MODEL PERFORMANCE ON UNSEEN TEST SET ---")
    print(f"Test Loss: {final_test_loss:.4f}")
    print(f"Test Accuracy: {final_test_accuracy:.4f}")
    
    # --- Print Raw Confusion Matrix Tensor ---
    print(f"\n--- Raw Confusion Matrix Tensor ---")
    print(final_confusion_matrix)

    # Print confusion matrix
    print("\nNormalized Confusion Matrix (Rows are True Labels, Columns are Predicted Labels):")
    for i, row in enumerate(final_confusion_matrix):
        true_class_name = CIFAR10_CLASSES[i] 
        
        # Check if the sum of the row is zero for  avoiding division by zero
        row_sum = row.sum()
        if row_sum > 0:
            normalized_row = row / row_sum
        else:
            normalized_row = torch.zeros_like(row) # If no true pixels for this class, all zeros

        print(f"True {true_class_name:<15}: ", end="")
        for j, val in enumerate(normalized_row):
            pred_class_name = CIFAR10_CLASSES[j] 
            print(f"{val.item()*100:>7.2f}% (Pred {pred_class_name:<10})", end="")
        print()
    
    print("\nTest completed successfully. Results printed above.")

    #To Run  use this in the bash after cd ProjectDirectory(Where the other source code files are stored):
    #Change the directory name
    # python test.py --model_path "/home/2f39/Computer_Vision_Raj/Checkpoints/lenet_adamW_checkpoint.pth" --data_root "/home/2f39/Computer_Vision_Raj/data" --num_classes 10 --batch_size 128 --num_workers 8
