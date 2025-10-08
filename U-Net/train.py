import os
import argparse
import random
import json 
import numpy as np
from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.classification import JaccardIndex
from torchmetrics import Accuracy 
from model import UNET 
from dataset import PascalVOCDataset 
from utils import (
    EarlyStopping, 
    set_seed, 
    save_checkpoint, 
    load_checkpoint, 
    get_metrics, 
    save_optimizer_config, 
    log_gpu_memory,
    
)

# --- Training and Validation Functions ---

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks.long()) # Ensure masks are long type

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer) # Unscale gradients before clipping their norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item(), "avg_loss": total_loss / (batch_idx + 1)})

    return total_loss / len(dataloader)

def validate_one_epoch(model, val_loader, criterion, device, num_classes, metrics_dict):
    #Validates the model for one epoch and computes metrics.
   
    model.eval()
    total_val_loss = 0.0
    
    # Reset metrics at the start of each validation epoch
    for metric in metrics_dict.values():
        metric.reset()

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(val_loader):
            data = data.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.long)

            predictions = model(data)
            loss = criterion(predictions, targets)

            total_val_loss += loss.item()

            preds_class_ids = torch.argmax(predictions, dim=1)
            
            # Update metrics with current batch's predictions and targets
            metrics_dict['mIoU'].update(preds_class_ids, targets)
            metrics_dict['pixel_acc'].update(preds_class_ids, targets)
            

    avg_val_loss = total_val_loss / len(val_loader)
    
    # Compute the final metrics for the entire validation set
    val_miou = metrics_dict['mIoU'].compute().item()
    val_pixel_accuracy = metrics_dict['pixel_acc'].compute().item()

    print(f"    [DEBUG - Val Epoch Summary] Average Validation Loss: {avg_val_loss:.6f}")
    print(f"    [DEBUG - Val Epoch Summary] Calculated mIoU: {val_miou:.4f}")
    print(f"    [DEBUG - Val Epoch Summary] Calculated Pixel Accuracy: {val_pixel_accuracy:.4f}")

    return avg_val_loss, val_miou, val_pixel_accuracy

def test_model(model, test_loader, criterion, device, num_classes, metrics_dict):
   #Evaluates the trained model on the test set 
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
    test_confusion_matrix = metrics_dict['conf_mat'].compute() #  confusion matrix

    return test_loss, test_miou, test_pixel_accuracy, test_confusion_matrix # Return confusion matrix


# --- Main Training Function ---
def train(model, train_loader, val_loader, optimizer, train_criterion, eval_criterion, device,
          epochs=30, checkpoint_path="checkpoints/checkpoint.pth", resume_checkpoint=False,
          patience=5, min_delta=0.0, num_classes=21, EARLY_STOPPING_WARMUP=5):

    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta, mode='min', verbose=True, warmup=EARLY_STOPPING_WARMUP) 
    
    best_val_loss = float("inf")
    start_epoch = 0
    train_losses, val_losses, val_mious = [], [], []
    val_pixel_accuracies = [] # Initialize as empty list for correct appending

    # Initialize metrics for validation 
    val_metrics = get_metrics(num_classes, device) 

    if resume_checkpoint and os.path.exists(checkpoint_path):
        print(f"Attempting to resume from {checkpoint_path}")
        checkpoint = load_checkpoint( 
            checkpoint_path, 
            model, 
            optimizer, 
            scaler, 
            early_stopper, 
            device
        )
        if checkpoint:
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            train_losses = checkpoint.get("train_losses", [])
            val_losses = checkpoint.get("val_losses", [])
            val_mious = checkpoint.get("val_mious", [])
            val_pixel_accuracies = checkpoint.get("val_pixel_accuracies", [])
            print(f"Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")
        else:
            print("Checkpoint loading failed or file not found. Starting training from scratch.")
            resume_checkpoint = False 
    else:
        print("Starting training from scratch.")
    
    log_gpu_memory() 

    for epoch in range(start_epoch, epochs):
        log_gpu_memory() 
        
        train_loss = train_one_epoch(model, train_loader, optimizer, train_criterion, scaler, device) 
        
        val_loss, val_miou, val_pixel_accuracy = validate_one_epoch(
            model, val_loader, eval_criterion, device, num_classes, val_metrics)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f} | Val Pixel Acc: {val_pixel_accuracy:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_mious.append(val_miou)
        val_pixel_accuracies.append(val_pixel_accuracy) # FIXED THIS BUG

        if early_stopper(val_loss, epoch): 
            print("Early stopping triggered.")
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Validation loss improved to {best_val_loss:.4f}. Saving checkpoint to {checkpoint_path}")
            save_checkpoint({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "early_stopper_state_dict": early_stopper.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_mious": val_mious,
                "val_pixel_accuracies": val_pixel_accuracies,
            }, filename=checkpoint_path)

    print("Training finished.")

    if os.path.exists(checkpoint_path):
        print("Loading best model weights from checkpoint for final state/testing.")
        load_checkpoint(checkpoint_path, model, device=device) 
    else:
        print("No checkpoint found to load best model weights. Using last epoch's weights for final plotting/testing.")
    return train_losses, val_losses, val_mious, val_pixel_accuracies


# --- Main execution block ---
if __name__ == "__main__":
    import albumentations as A 
    from albumentations.pytorch import ToTensorV2 

    SEED = 42
    set_seed(SEED) 
    print(f"DEBUG: Random seed set to {SEED}")
    parser = argparse.ArgumentParser(description="Train a U-Net for Pascal VOC Segmentation")
    parser.add_argument("--data_dir", type=str, default="/home/2f39/Computer_Vision_Raj/data/VOCdevkit/VOC2012")
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=21)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--checkpoint_path", type=str, default="Checkpoints/checkpoint.pth",
                         help="Path to save/load model checkpoint. Default: checkpoints/checkpoint.pth")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--es_warmup_epochs", type=int, default=5, 
                         help="Number of initial epochs to disable early stopping warmup.")
   
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Transforms ---
    train_transform = A.Compose([
        A.Resize(height=args.image_height, width=args.image_width),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        
        A.Normalize(mean=[0.45677424265616295, 0.443102272306289, 0.4082499674586274],
            std=[0.2369761136780325, 0.2332828798308419, 0.23898276282840822],
            max_pixel_value=255.0),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})

    val_transform = A.Compose([
        A.Resize(height=args.image_height, width=args.image_width),
        A.Normalize(mean=[0.45677424265616295, 0.443102272306289, 0.4082499674586274],
            std=[0.2369761136780325, 0.2332828798308419, 0.23898276282840822],
            max_pixel_value=255.0),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})

    image_dir = os.path.join(args.data_dir, "JPEGImages")
    mask_dir = os.path.join(args.data_dir, "SegmentationClass")
    train_split_file = os.path.join(args.data_dir, "ImageSets", "Segmentation", "train.txt")
    val_split_file = os.path.join(args.data_dir, "ImageSets", "Segmentation", "val.txt")
    test_split_file = os.path.join(args.data_dir, "ImageSets", "Segmentation", "test.txt")

    train_dataset = PascalVOCDataset(image_dir, mask_dir, train_split_file, transform=train_transform)
    val_dataset = PascalVOCDataset(image_dir, mask_dir, val_split_file, transform=val_transform)
    test_dataset = PascalVOCDataset(image_dir, mask_dir, test_split_file, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print(f"Train dataset size: {len(train_dataset)} samples")
    print(f"Val dataset size: {len(val_dataset)} samples")
    print(f"Test dataset size: {len(test_dataset)} samples")

    model = UNET(in_channels=3, out_channels=args.num_classes).to(device)

    class_counts = np.array([
        182014429,  # 00 background
        1780580,    # 01 aeroplane
        758311,     # 02 bicycle
        2232247,    # 03 bird
        1514260,    # 04 boat
        1517186,    # 05 bottle
        4375622,    # 06 bus
        3494749,    # 07 car
        6752515,    # 08 cat
        2861091,    # 09 chair
        2060925,    # 10 cow
        3381632,    # 11 diningtable
        4344951,    # 12 dog
        2283739,    # 13 horse
        2888641,    # 14 motorbike
        11995853,   # 15 person
        1670340,    # 16 pottedplant
        2254463,    # 17 sheep
        3612229,    # 18 sofa
        3984238,    # 19 train
        2349235     # 20 tvmonitor
    ], dtype=np.float32)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.mean() # Normalize weights
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    train_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, ignore_index=255)
    eval_criterion = nn.CrossEntropyLoss(ignore_index=255)

    optimizer = optim.AdamW(   ### <-SGD,SGD with Momentum, RMSprop and Adam will replace
        model.parameters(),
        lr= 0.0012520139985656086,
        betas= (0.8924233488990823,0.9813567499188975),
        eps=7.865391378834923e-34,
        weight_decay= 0.000521042497816867)

    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    optimizer_config_path = os.path.join(os.path.dirname(args.checkpoint_path), f"{type(optimizer).__name__}_config.json")
    save_optimizer_config(optimizer, optimizer_config_path)

    print("Starting training process...")
    train_losses, val_losses, val_mious, val_pixel_accuracies = train( 
        model,
        train_loader,
        val_loader,
        optimizer,
        train_criterion,
        eval_criterion,
        device,
        epochs=args.epochs,
        checkpoint_path=args.checkpoint_path,
        resume_checkpoint=args.resume,
        patience=args.patience,
        min_delta=args.min_delta,
        num_classes=args.num_classes,
        es_warmup_epochs=args.es_warmup_epochs
    )

    print(f"\n--- Running Final Evaluation on Test Set ---")
    
    test_metrics = get_metrics(args.num_classes, device)
    
    #Get confusion matrix from test_model
    final_test_loss, final_test_miou, final_test_pixel_accuracy, final_confusion_matrix = test_model(
        model, test_loader, eval_criterion, device, 
        num_classes=args.num_classes, metrics_dict=test_metrics
    )

    print(f"\n--- FINAL MODEL PERFORMANCE ON UNSEEN TEST SET ---")
    print(f"Test Loss: {final_test_loss:.4f}")
    print(f"Test mIoU: {final_test_miou:.4f}")
    print(f"Test Pixel Accuracy: {final_test_pixel_accuracy:.4f}")
    
    #  Pascal VOC class labels for interpreting confusion matrix
    pascal_voc_class_labels = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
        "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]

    # --- Print Raw Confusion Matrix Tensor ---
    print(f"\n--- Raw Confusion Matrix Tensor ---")
    print(final_confusion_matrix)

    # print a more readable confusion matrix
    print("\nRaw Confusion Matrix (with labels, normalized by true class):")
    for i, row in enumerate(final_confusion_matrix):
        print(f"True {pascal_voc_class_labels[i]:<15}: ", end="")
        for j, val in enumerate(row):
            # Print as percentage with 2 decimal places for better readability of normalized values
            print(f"{val.item()*100:>7.2f}% (Pred {pascal_voc_class_labels[j]:<10})", end="") 
        print()

    print("\nAll training, validation, and testing steps completed successfully. Results printed above.")

    # --- : Saving Raw Training History and Confusion Matrix Data ---
    output_data_dir = os.path.join(os.path.dirname(args.checkpoint_path), "run_data")
    os.makedirs(output_data_dir, exist_ok=True)
    
    # Save training history to a JSON file
    history_data = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_mious": val_mious,
        "val_pixel_accuracies": val_pixel_accuracies
    }
    history_save_path = os.path.join(output_data_dir, f"{type(optimizer).__name__}_training_history.json")
    with open(history_save_path, 'w') as f:
        json.dump(history_data, f)
    print(f"\nTraining history saved to: {history_save_path}")

    # Saving confusion matrix tensor to a PyTorch .pt file
    
    confusion_matrix_save_path_pt = os.path.join(output_data_dir, f"{type(optimizer).__name__}_confusion_matrix.pt")
    torch.save(final_confusion_matrix.cpu(), confusion_matrix_save_path_pt)
    print(f"Confusion matrix tensor saved to: {confusion_matrix_save_path_pt}")

