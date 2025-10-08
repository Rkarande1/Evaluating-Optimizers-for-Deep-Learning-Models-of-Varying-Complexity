# Imports
import os
import random
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

# Local imports
from model import LeNet5
from utils import (
    EarlyStopping, 
    set_seed, 
    get_metrics,
    get_cifar10_dataloaders,
    save_checkpoint, 
    load_checkpoint, 
    save_optimizer_config,
    plot_single_run_metrics
)

# --- Training and Validation Functions ---

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch_num, total_epochs):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_num}/{total_epochs} Training", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            logits, probas = model(images)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()}) # Simplified postfix

    return total_loss / len(dataloader)

def validate_one_epoch(model, val_loader, criterion, device, num_classes, metrics_dict, epoch_num, total_epochs):
    model.eval()
    total_val_loss = 0.0
    metrics_dict['accuracy'].reset()

    pbar = tqdm(val_loader, desc=f"Epoch {epoch_num}/{total_epochs} Validation", leave=False)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            logits, probas = model(images)
            loss = criterion(logits, labels)

            total_val_loss += loss.item()
            
            predicted_labels = torch.argmax(probas, dim=1)
            metrics_dict['accuracy'].update(predicted_labels, labels)

            pbar.set_postfix({"val_loss": loss.item()}) # Simplified postfix

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = metrics_dict['accuracy'].compute().item()

    print(f"    Val Loss: {avg_val_loss:.6f} | Val Acc: {val_accuracy:.4f}")

    return avg_val_loss, val_accuracy

def test_model(model, test_loader, criterion, device, num_classes, metrics_dict):
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


# --- Main Training Function ---
def train_model(model, train_loader, val_loader, test_loader, optimizer, train_criterion, eval_criterion, device,
                epochs, checkpoint_path, resume_checkpoint, patience, min_delta, num_classes, es_warmup_epochs):
    
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta, mode='min', verbose=True, warmup=es_warmup_epochs) 
    
    best_val_loss = float("inf")
    start_epoch = 0
    train_losses, val_losses, val_accuracies = [], [], []

    val_metrics = get_metrics(num_classes, device) 
    test_metrics = get_metrics(num_classes, device)

    if resume_checkpoint and os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        checkpoint = load_checkpoint( 
            checkpoint_path, model, optimizer, scaler, early_stopper, device
        )
        if checkpoint:
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            train_losses = checkpoint.get("train_losses", [])
            val_losses = checkpoint.get("val_losses", [])
            val_accuracies = checkpoint.get("val_accuracies", [])
            print(f"Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")
        else:
            print("Checkpoint loading failed. Starting from scratch.")
            resume_checkpoint = False 
    else:
        print("Starting from scratch.")
    
    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, train_criterion, scaler, device, epoch + 1, epochs) 
        val_loss, val_accuracy = validate_one_epoch(
            model, val_loader, eval_criterion, device, num_classes, val_metrics, epoch + 1, epochs)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if early_stopper(val_loss, epoch): 
            print("Early stopping triggered.")
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Validation loss improved. Saving checkpoint to {checkpoint_path}")
            save_checkpoint({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "early_stopper_state_dict": early_stopper.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_accuracies": val_accuracies,
            }, filename=checkpoint_path)

    print("Training finished.")
    
    if os.path.exists(checkpoint_path):
        print("Loading best model weights for final testing.")
        load_checkpoint(checkpoint_path, model, device=device) 
    else:
        print("No checkpoint found. Using last epoch's weights.")

    final_test_loss, final_test_accuracy, final_confusion_matrix = test_model(
        model, test_loader, eval_criterion, device, num_classes, test_metrics
    )

    print(f"\n--- Final Test Set Performance ---")
    print(f"Test Loss: {final_test_loss:.4f}")
    print(f"Test Accuracy: {final_test_accuracy:.4f}")
    
    output_data_dir = os.path.join(os.path.dirname(checkpoint_path), "run_data")
    os.makedirs(output_data_dir, exist_ok=True)
    
    history_data = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "final_test_loss": final_test_loss,
        "final_test_accuracy": final_test_accuracy
    }
    history_save_path = os.path.join(output_data_dir, f"{type(optimizer).__name__}_training_history.json")
    with open(history_save_path, 'w') as f:
        json.dump(history_data, f, indent=4)
    print(f"History saved to: {history_save_path}")

    confusion_matrix_save_path_pt = os.path.join(output_data_dir, f"{type(optimizer).__name__}_confusion_matrix.pt")
    torch.save(final_confusion_matrix.cpu(), confusion_matrix_save_path_pt)
    print(f"Confusion matrix saved to: {confusion_matrix_save_path_pt}")

    plot_save_dir = os.path.join(os.path.dirname(checkpoint_path), "plots")
    os.makedirs(plot_save_dir, exist_ok=True)

    plot_single_run_metrics(
        train_losses, val_losses, val_accuracies, 
        final_confusion_matrix, save_dir=plot_save_dir, 
        optimizer_name=type(optimizer).__name__, 
        epoch_offset=start_epoch,
        num_classes=num_classes
    )
    print("Training and testing complete.")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "final_test_loss": final_test_loss,
        "final_test_accuracy": final_test_accuracy,
        "final_confusion_matrix": final_confusion_matrix
    }


if __name__ == "__main__":
    # --- Configuration Constants ---
    DATA_ROOT = "data"
    EPOCHS = 50
    BATCH_SIZE = 128
    NUM_CLASSES = 10
    CHECKPOINT_PATH = "/home/2f39/Computer_Vision_Raj/Checkpoints/lenet_adamW_checkpoint.pth" 
    RESUME_TRAINING = False
    PATIENCE = 5
    MIN_DELTA = 0.001
    ES_WARMUP_EPOCHS = 3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(42)
    
    # --- Data Loaders ---
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        batch_size=BATCH_SIZE,
        data_root=DATA_ROOT
    )

    print(f"Train/Val/Test samples: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")

    model = LeNet5(num_classes=NUM_CLASSES, in_channels=3).to(device)

    train_criterion = nn.CrossEntropyLoss()
    eval_criterion = nn.CrossEntropyLoss()

    # - Optimizer Definition (Hardcoded) -

    optimizer = optim.AdamW( #<-----Changing this block for each optimizer
        model.parameters(),
        lr= 0.0015294334405757852,
        betas= (0.8857338434453298,0.9866173584306264),
        eps=8.232712072446393e-26,
        weight_decay= 0.006746005946737532)
    

    # Save optimizer config
    optimizer_config_output_dir = os.path.dirname(CHECKPOINT_PATH)
    optimizer_config_path = os.path.join(optimizer_config_output_dir, f"{type(optimizer).__name__}_config.json")
    save_optimizer_config(optimizer, optimizer_config_path)

    print(f"Starting training with {type(optimizer).__name__}...")
    final_results = train_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        train_criterion,
        eval_criterion,
        device,
        epochs=EPOCHS,
        checkpoint_path=CHECKPOINT_PATH,
        resume_checkpoint=RESUME_TRAINING,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        num_classes=NUM_CLASSES,
        es_warmup_epochs=ES_WARMUP_EPOCHS
    )
    
    print("\n--- Run Complete ---")
    print(f"Final Test Loss: {final_results['final_test_loss']:.4f}")
    print(f"Final Test Accuracy: {final_results['final_test_accuracy']:.4f}")

