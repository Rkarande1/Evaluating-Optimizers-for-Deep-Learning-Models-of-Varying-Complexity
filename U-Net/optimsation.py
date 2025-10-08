# imports
import os
import random
import numpy as np
import optuna
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import UNET
from dataset import PascalVOCDataset
from train import train_one_epoch, validate_one_epoch

from utils import (
    EarlyStopping,
    set_seed,
    get_metrics, # Get TorchMetrics instances
    plot_losses, # Plot single trial performance
    plot_combined_optimizer_metrics # Plot all optimizers together
)

# --- Configuration Constants ---
DATA_ROOT = "/content/data/VOCdevkit/VOC2012"
IMAGE_DIR = os.path.join(DATA_ROOT, "JPEGImages")
MASK_DIR = os.path.join(DATA_ROOT, "SegmentationClass")
TRAIN_SPLIT_FILE = os.path.join(DATA_ROOT, "ImageSets", "Segmentation", "train.txt")
VAL_SPLIT_FILE = os.path.join(DATA_ROOT, "ImageSets", "Segmentation", "val.txt")

NUM_CLASSES = 21
EPOCHS_PER_TRIAL = 20 # Epochs for each Optuna trial
TRIALS_PER_OPTIMIZER = 10 # Trials per optimizer
EARLY_STOPPING_WARMUP = 3 # Early stopping warmup epochs

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Storage paths
STUDY_STORAGE_BASE = "/content/drive/MyDrive/Raj/optuna_studies"
os.makedirs(STUDY_STORAGE_BASE, exist_ok=True)
PLOT_OUTPUT_DIR = os.path.join(STUDY_STORAGE_BASE, "plots")
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)


# --- Optuna Objective Function ---
def create_objective_for_optimizer(optimizer_type):

    def objective(trial):
        torch.cuda.empty_cache() # Clear GPU memory
        set_seed(42)
        batch_size = trial.suggest_categorical("batch_size", [128])

        model = UNET(in_channels=3, out_channels=NUM_CLASSES).to(device)

        # --- Class Weighting ---
        class_counts = np.array([
            182014429,   # 00 background
            1780580,     # 01 aeroplane
            758311,      # 02 bicycle
            2232247,     # 03 bird
            1514260,     # 04 boat
            1517186,     # 05 bottle
            4375622,     # 06 bus
            3494749,     # 07 car
            6752515,     # 08 cat
            2861091,     # 09 chair
            2060925,     # 10 cow
            3381632,     # 11 diningtable
            4344951,     # 12 dog
            2283739,     # 13 horse
            2888641,     # 14 motorbike
            11995853,    # 15 person
            1670340,     # 16 pottedplant
            2254463,     # 17 sheep
            3612229,     # 18 sofa
            3984238,     # 19 train
            2349235      # 20 tvmonitor
        ], dtype=np.float32)

        # Calculate inverse frequency weights
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.mean() # Normalize
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, ignore_index=255)

        # Initialize optimizer based on type
        if optimizer_type == "Adam":
            lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
            beta1 = trial.suggest_float("adam_beta1", 0.87, 0.9)
            beta2 = trial.suggest_float("adam_beta2", 0.98, 0.999)
            eps = trial.suggest_float("adam_eps", 1e-8, 1e-6, log=True)
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)

        elif optimizer_type == "AdamW":
            lr = trial.suggest_float("lr", 0.001, 0.01, log=True)
            beta1 = trial.suggest_float("adamw_beta1", 0.88, 0.9)
            beta2 = trial.suggest_float("adamw_beta2", 0.98, 0.999)
            eps = trial.suggest_float("adamw_eps", 1e-82, 1e-6, log=True)
            weight_decay = trial.suggest_float("adamw_weight_decay", 1e-5, 1e-2, log=True)
            optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)

        elif optimizer_type == "SGD": # SGD without momentum
            lr = trial.suggest_float("lr", 1e-3, 0.1, log=True)
            optimizer = optim.SGD(model.parameters(), lr=lr)

        elif optimizer_type == "SGD_Momentum": # SGD with momentum
            lr = trial.suggest_float("lr", 1e-3, 0.1, log=True)
            momentum = trial.suggest_float("momentum", 0.8, 0.99)
            weight_decay = trial.suggest_float("sgd_momentum_weight_decay", 1e-5, 1e-3, log=True)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        elif optimizer_type == "RMSprop":
            lr = trial.suggest_float("lr", 0.001, 0.01, log=False)
            alpha = trial.suggest_float("rmsprop_alpha", 0.9, 0.99) # Smoothing constant
            eps = trial.suggest_float("rmsprop_eps", 1e-8, 1e-6, log=True)
            momentum = trial.suggest_float("rmsprop_momentum", 0.0, 0.6) # Can be 0 momentum
            weight_decay = trial.suggest_float("rmsprop_weight_decay", 1e-5, 1e-3, log=True)
            optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, eps=eps, momentum=momentum, weight_decay=weight_decay)

        else:
            raise ValueError(f"Unknown optimizer_type: {optimizer_type}")

        # Data transforms
        transform = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.4),
            A.Normalize(mean=[0.45677424265616295, 0.443102272306289, 0.4082499674586274],
            std=[0.2369761136780325, 0.2332828798308419, 0.23898276282840822],
            max_pixel_value=255.0),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})

        train_dataset = PascalVOCDataset(IMAGE_DIR, MASK_DIR, TRAIN_SPLIT_FILE, transform=transform)
        val_dataset = PascalVOCDataset(IMAGE_DIR, MASK_DIR, VAL_SPLIT_FILE, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
        # Early stopping setup
        early_stopper = EarlyStopping(patience=5, min_delta=0.001, verbose=False, warmup=EARLY_STOPPING_WARMUP)

        trial_train_losses = []
        trial_val_losses = []
        trial_val_mious = []
        trial_val_pixel_accuracies = []

        best_val_loss_current_trial = float("inf")

        # Initialize torchmetrics
        trial_metrics = get_metrics(NUM_CLASSES, device)

        for epoch in range(EPOCHS_PER_TRIAL):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
            val_loss, val_miou, val_pixel_accuracy = validate_one_epoch(model, val_loader, criterion, device, NUM_CLASSES, trial_metrics)

            trial_train_losses.append(train_loss)
            trial_val_losses.append(val_loss)
            trial_val_mious.append(val_miou)
            trial_val_pixel_accuracies.append(val_pixel_accuracy)

            # Check early stopping
            early_stop_triggered = early_stopper(val_loss, epoch)
            if early_stop_triggered:
                print(f"Trial {trial.number} for {optimizer_type}: Early stopping triggered at epoch {epoch+1} (val_loss: {val_loss:.4f}).")
                break

            if val_loss < best_val_loss_current_trial:
                best_val_loss_current_trial = val_loss

        # Print trial summary
        print(f"Trial {trial.number} for {optimizer_type} - "
              f"LR: {lr:.2e}, BS: {batch_size}, "
              f"Final Val Loss: {best_val_loss_current_trial:.4f}")

        # Store metrics
        trial.set_user_attr("train_losses", trial_train_losses)
        trial.set_user_attr("val_losses", trial_val_losses)
        trial.set_user_attr("val_mious", trial_val_mious)
        trial.set_user_attr("val_pixel_accuracies", trial_val_pixel_accuracies)
        trial.set_user_attr("optimizer_name", optimizer_type) # Store optimizer name

        return best_val_loss_current_trial # Return value to minimize

    return objective

# --- Callback to print trial progress ---
def print_progress_callback(study, trial):
    """Prints progress for each Optuna trial."""
    if trial.state == optuna.trial.TrialState.COMPLETE:
        print(f"   Trial {trial.number}: Status: {trial.state.name}, Value: {trial.value:.4f}, Params: {trial.params}")
    elif trial.state == optuna.trial.TrialState.PRUNED:
        print(f"   Trial {trial.number}: Status: {trial.state.name}")


if __name__ == "__main__":
    # Optimizers to compare
    optimizer_types_to_compare = ["SGD_Momentum","AdamW","Adam","RMSprop","SGD"]
    all_best_trials = {} # Store best trial for each optimizer

    for opt_type in optimizer_types_to_compare:
        print(f"\n--- Running Optuna Study for {opt_type} ---")
        # Define study storage
        study_name = f"pascal_voc_unet_study_{opt_type.lower()}"
        storage_path = os.path.join(STUDY_STORAGE_BASE, f"{study_name}.db")

        # Create or load Optuna study
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=f"sqlite:///{storage_path}",
            load_if_exists=True
        )

        # Handle resuming studies
        num_completed_trials = len(study.trials)
        remaining_trials = TRIALS_PER_OPTIMIZER - num_completed_trials

        if remaining_trials <= 0:
            print(f"Study '{study_name}' already complete with {num_completed_trials} trials.")
            print(f"Skipping new optimization for {opt_type}.")
        else:
            print(f"Study '{study_name}' has {num_completed_trials} trials completed. Running {remaining_trials} more.")
            study.optimize(
                create_objective_for_optimizer(opt_type),
                n_trials=remaining_trials, # Run remaining trials
                gc_after_trial=True,
                callbacks=[print_progress_callback]
            )

        # Print best trial info
        if len(study.trials) > 0 and study.best_trial is not None:
            print(f"\nStudy finished for {opt_type}. Best trial:")
            print(f"   Value: {study.best_value:.4f}")
            print(f"   Params: {study.best_params}")
            all_best_trials[opt_type] = study.best_trial
        else:
            print(f"\nNo successful trials found for {opt_type} in study '{study_name}'.")

    print("\n--- All Optimizer Studies Complete ---")

    # Generate individual plots for best trial of each optimizer
    for opt_type, best_trial_for_opt in all_best_trials.items():
        train_losses = best_trial_for_opt.user_attrs["train_losses"]
        val_losses = best_trial_for_opt.user_attrs["val_losses"]
        val_mious = best_trial_for_opt.user_attrs["val_mious"]
        val_pixel_accuracies = best_trial_for_opt.user_attrs["val_pixel_accuracies"]
        optimizer_name = best_trial_for_opt.user_attrs["optimizer_name"]

        # Define individual plot filename
        plot_filename = os.path.join(PLOT_OUTPUT_DIR, f"{optimizer_name}_best_trial_performance.png")

        # Call plotting function
        plot_losses(
            train_losses,
            val_losses,
            val_mious,
            val_pixel_accuracies,
            save_path=plot_filename,
            optimizer_name=optimizer_name,
            epoch_offset=0 #optuna starts from epoch zero
        )
        print(f"Individual plot for {optimizer_name} (best trial) saved to {plot_filename}")

    # Collect data for combined plot
    combined_plot_data = []
    for opt_type, best_trial_for_opt in all_best_trials.items():
        combined_plot_data.append({
            'optimizer_name': opt_type,
            'val_mious': best_trial_for_opt.user_attrs["val_mious"],
            'val_pixel_accuracies': best_trial_for_opt.user_attrs["val_pixel_accuracies"]
        })

    # Combined plot filename
    combined_plot_filename = os.path.join(PLOT_OUTPUT_DIR, "combined_optimizer_performance.png")

    plot_combined_optimizer_metrics(combined_plot_data, save_path=combined_plot_filename)
    print(f"Combined optimizer metrics plot saved to {combined_plot_filename}")

    print("\nOptimization process complete. Check the plots for performance comparison.")