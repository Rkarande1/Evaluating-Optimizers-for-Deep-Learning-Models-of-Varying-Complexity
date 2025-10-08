import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Assuming model.py, dataset.py, and utils.py are in the same directory
from model import UNET
from dataset import PascalVOCDataset
from utils import load_checkpoint, get_metrics, PASCAL_VOC_CLASSES, set_seed # Import only necessary utils

# --- Test Function (Copied directly from your train.py) ---
def test_model(model, test_loader, criterion, device, num_classes, metrics_dict):
    """
    Evaluates the trained model on the test set.
    """
    model.eval()
    test_loss = 0.0
    
    # Reset metrics for the test run
    for metric in metrics_dict.values():
        metric.reset()

    print("\n--- Starting Final Evaluation on Test Set ---")
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks.long())
            test_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            metrics_dict['mIoU'].update(preds, masks)
            metrics_dict['pixel_acc'].update(preds, masks)
            metrics_dict['conf_mat'].update(preds, masks) 

    test_loss /= len(test_loader)
    test_miou = metrics_dict['mIoU'].compute().item()
    test_pixel_accuracy = metrics_dict['pixel_acc'].compute().item()
    test_confusion_matrix = metrics_dict['conf_mat'].compute() # confusion matrix

    return test_loss, test_miou, test_pixel_accuracy, test_confusion_matrix # Return confusion matrix


# --- Main execution block for test.py ---
if __name__ == "__main__":
    SEED=42
    set_seed(SEED)
    parser = argparse.ArgumentParser(description="Test a Semantic Segmentation Model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="checkpoint_adamW_final_TEST.pth") ##<-- Paste the checkpoint/best model path e.g checkpoint_sgd_final_TEST.pth
                                                                        #Thats inside checkpoint folder 
    parser.add_argument("--data_dir", type=str, default="./data/VOCdevkit/VOC2012",
                        help="Root directory for Pascal VOC dataset.")
    parser.add_argument("--image_height", type=int, default=256, help="Image height for model input.")
    parser.add_argument("--image_width", type=int, default=256, help="Image width for model input.")
    parser.add_argument("--num_classes", type=int, default=21, help="Number of segmentation classes (e.g., 21 for VOC).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for testing.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loader workers.") # Changed to 8 as in your train.py
    parser.add_argument("--ignore_index", type=int, default=255, help="Pixel value to ignore in metrics (e.g., void class).")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Transforms (same as your validation transform) ---
    test_transform = A.Compose([
        A.Resize(height=args.image_height, width=args.image_width),
        A.Normalize(mean=[0.45677424265616295, 0.443102272306289, 0.4082499674586274],
                    std=[0.2369761136780325, 0.2332828798308419, 0.23898276282840822],
                    max_pixel_value=255.0),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})

    # --- Dataset and DataLoader ---
    image_dir = os.path.join(args.data_dir, "JPEGImages")
    mask_dir = os.path.join(args.data_dir, "SegmentationClass")
    test_split_file = os.path.join(args.data_dir, "ImageSets", "Segmentation", "test.txt") # Use test.txt for final evaluation

    test_dataset = PascalVOCDataset(image_dir, mask_dir, test_split_file, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"Test dataset size: {len(test_dataset)} samples")

    # --- Model Initialization ---
    model = UNET(in_channels=3, out_channels=args.num_classes).to(device)

    # --- Load Model Checkpoint ---
    checkpoint_loaded = load_checkpoint(args.model_path, model, device=device)
    if not checkpoint_loaded:
        print("Failed to load model checkpoint. Please ensure the path is correct and the file exists.")
        exit() # Exit if model cannot be loaded

    # --- Loss Function for calculating test loss ---
    # Using eval_criterion from your train.py setup
    eval_criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)

    # --- Initialize Metrics for Testing ---
    test_metrics = get_metrics(args.num_classes, device, ignore_index=args.ignore_index)
    
    # --- Run Final Test ---
    final_test_loss, final_test_miou, final_test_pixel_accuracy, final_confusion_matrix = test_model(
        model, test_loader, eval_criterion, device, 
        num_classes=args.num_classes, metrics_dict=test_metrics
    )

    print(f"\n--- FINAL MODEL PERFORMANCE ON UNSEEN TEST SET ---")
    print(f"Test Loss: {final_test_loss:.4f}")
    print(f"Test mIoU: {final_test_miou:.4f}")
    print(f"Test Pixel Accuracy: {final_test_pixel_accuracy:.4f}")
    
    # --- Print Raw Confusion Matrix Tensor ---
    print(f"\n--- Raw Confusion Matrix Tensor ---")
    print(final_confusion_matrix)

    # Pascal VOC class labels for interpreting confusion matrix
    pascal_voc_class_labels = PASCAL_VOC_CLASSES # Assuming this is imported from utils

    # Print a more readable confusion matrix (normalized by true class as in your train.py)
    print("\nRaw Confusion Matrix (with labels, normalized by true class):")
    for i, row in enumerate(final_confusion_matrix):
        true_class_name = pascal_voc_class_labels[i]
        
        # Check if the sum of the row is zero to avoid division by zero
        row_sum = row.sum()
        if row_sum > 0:
            normalized_row = row / row_sum
        else:
            normalized_row = torch.zeros_like(row) # If no true pixels for this class, all zeros

        print(f"True {true_class_name:<15}: ", end="")
        for j, val in enumerate(normalized_row):
            pred_class_name = pascal_voc_class_labels[j]
            print(f"{val.item()*100:>7.2f}% (Pred {pred_class_name:<10})", end="")
        print()
    
    print("\nTest completed successfully. Results printed above.")

    #IN BASH:
    #python test.py \
    #--model_path "C:/Users/User/Project/Checkpoints/checkpoint_sgd_final_TEST.pth" \
    #--data_dir "C:/Users/User/Project/data/VOCdevkit/VOC2012" \
    #--image_height 256 \
    #--image_width 256 \
    #--num_classes 21 \
    #--batch_size 128 \
    #--num_workers 8