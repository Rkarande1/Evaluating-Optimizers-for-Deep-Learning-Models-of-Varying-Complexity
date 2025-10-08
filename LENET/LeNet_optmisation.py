# Standard library imports
import os
import random
import json

# Third-party imports
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # For progress bars

# Local application imports
from model import LeNet5 # Import the LeNet5 model
from utils import (
    EarlyStopping, 
    set_seed, 
    get_metrics, # For initializing torchmetrics (accuracy, confusion matrix)
    get_cifar10_dataloaders, # For loading CIFAR-10 data with train/val split
    log_gpu_memory,
    plot_combined_optimizer_metrics # For combined plots of best trials
)

# --- Configuration Constants ---
# You'll need to set up your Google Drive mount in Colab for persistent storage
# Example: !mkdir -p /content/drive/MyDrive/Raj/optuna_studies
STUDY_STORAGE_BASE = "/home/2f39/Computer_Vision_Raj/optuna_studies" # Directory to store study databases
os.makedirs(STUDY_STORAGE_BASE, exist_ok=True) # Ensure study directory exists

# Number of epochs each Optuna trial will run for
# Set a reasonable number, early stopping will prevent overtraining
EPOCHS_PER_TRIAL = 30 
# Number of optimization trials for each optimizer type
TRIALS_PER_OPTIMIZER = 10 
# Number of initial epochs to disable early stopping warmup.
EARLY_STOPPING_WARMUP = 5 
# Patience for early stopping during Optuna trials
EARLY_STOPPING_PATIENCE = 7 
# Minimum delta for early stopping (e.g., 0.001 for loss)
EARLY_STOPPING_MIN_DELTA = 0.001 

# Global device definition
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Constants for CIFAR-10
NUM_CLASSES = 10
BATCH_SIZE_OPTIMIZATION = 128 # Fixed batch size for optimization trials
TRAIN_VAL_SPLIT_RATIO = 0.9 # 45k train, 5k val from 50k CIFAR-10 train set

# --- Optuna Objective Function ---
def create_objective_for_optimizer(optimizer_type):
    """
    Returns an objective function tailored for a specific optimizer_type.
    This function will be passed to Optuna's study.optimize().
    """
    def objective(trial):
        torch.cuda.empty_cache() # Clear GPU cache to reduce memory issues between trials
        
        # Seed for reproducibility for THIS specific trial
        # Using a fixed seed (e.e.g., 42) makes comparisons between optimizers more direct.
        set_seed(42) 
        
        model = LeNet5(num_classes=NUM_CLASSES, in_channels=3).to(DEVICE)
        criterion = nn.CrossEntropyLoss() # Standard Cross-Entropy Loss for classification

        # --- Hyperparameters to search based on optimizer_type ---
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
            momentum = trial.suggest_float("rmsprop_momentum", 0.0, 0.6) # Can be 0 for no momentum
            weight_decay = trial.suggest_float("rmsprop_weight_decay", 1e-5, 1e-3, log=True)
            optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, eps=eps, momentum=momentum, weight_decay=weight_decay)

        else:
            raise ValueError(f"Unknown optimizer_type: {optimizer_type}")

        # --- Data Loaders ---
        # Get data loaders with custom train/val split
        train_loader, val_loader, _ = get_cifar10_dataloaders(
            batch_size=BATCH_SIZE_OPTIMIZATION, 
            train_val_split_ratio=TRAIN_VAL_SPLIT_RATIO, 
            num_workers=8
        )

        scaler = torch.amp.GradScaler(enabled=DEVICE.type == 'cuda')
        early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA, 
                                      mode='min', verbose=False, warmup=EARLY_STOPPING_WARMUP) 
        
        trial_train_losses = []
        trial_val_losses = []
        trial_val_accuracies = []

        best_val_loss_current_trial = float("inf")
        best_val_accuracy_current_trial = 0.0 # Track best accuracy

        # Initialize torchmetrics for this trial's validation
        trial_metrics = get_metrics(NUM_CLASSES, DEVICE)

        for epoch in range(EPOCHS_PER_TRIAL):
            # --- Training Loop ---
            model.train()
            total_train_loss = 0.0
            for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                with torch.amp.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == 'cuda'):
                    logits, probas = model(images)
                    loss = criterion(logits, labels)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
            trial_train_losses.append(avg_train_loss)

            # --- Validation Loop ---
            model.eval()
            total_val_loss = 0.0
            trial_metrics['accuracy'].reset() # Reset accuracy for current validation epoch
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    logits, probas = model(images)
                    loss = criterion(logits, labels)
                    total_val_loss += loss.item()
                    
                    # Update accuracy metric
                    predicted_labels = torch.argmax(probas, dim=1)
                    trial_metrics['accuracy'].update(predicted_labels, labels)

            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = trial_metrics['accuracy'].compute().item() # Compute accuracy for the epoch

            trial_val_losses.append(avg_val_loss)
            trial_val_accuracies.append(val_accuracy)

            
            trial.report(avg_val_loss, epoch)

            # Update best scores for current trial
            if avg_val_loss < best_val_loss_current_trial:
                best_val_loss_current_trial = avg_val_loss
            if val_accuracy > best_val_accuracy_current_trial:
                best_val_accuracy_current_trial = val_accuracy

            # Early stopping check
            if early_stopper(avg_val_loss, epoch): 
                print(f"Trial {trial.number} ({optimizer_type}): Early stopping triggered at epoch {epoch+1} (Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}).")
                break
            
            # it will be pruned if a trial is bad
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Store these metrics  for plotting later
        trial.set_user_attr("train_losses", trial_train_losses)
        trial.set_user_attr("val_losses", trial_val_losses)
        trial.set_user_attr("val_accuracies", trial_val_accuracies)
        trial.set_user_attr("optimizer_name", optimizer_type) # Store the fixed optimizer name

        # Optuna minimizes this value (validation loss)
        return best_val_loss_current_trial 
    
    return objective

# --- Callback for printing trial progress in the console  ---
def print_progress_callback(study, trial):
    """Callback to print progress for each trial (for optuna.optimize's callbacks)."""
    if trial.state == optuna.trial.TrialState.COMPLETE:
        print(f"  Trial {trial.number}: Status: {trial.state.name}, Value: {trial.value:.4f}, Params: {trial.params}")
    elif trial.state == optuna.trial.TrialState.PRUNED:
        print(f"  Trial {trial.number}: Status: {trial.state.name}")


if __name__ == "__main__":
 
    optimizers_to_compare = ["Adam", "AdamW", "SGD", "SGD_Momentum", "RMSprop"]
    
    all_best_trials_data = {} # Dictionary to store relevant data of the best trial for each optimizer

    for opt_type in optimizers_to_compare:
        print(f"\n--- Running Optuna Study for {opt_type} ---")
       
        study_name_for_db = f"cifar10_lenet_study_{opt_type}"
        storage_path = f"sqlite:///{os.path.join(STUDY_STORAGE_BASE, study_name_for_db)}.db"
        
        # Create or load an Optuna study for the current optimizer type
        study = optuna.create_study(
            direction="minimize",     
            study_name=study_name_for_db, 
            storage=storage_path,    
            load_if_exists=True        # Load existing study if it already exists
        )
        
        # Determine how many trials are left to run
        num_completed_trials = len(study.trials)
        remaining_trials = TRIALS_PER_OPTIMIZER - num_completed_trials

        if remaining_trials <= 0:
            print(f"Study '{study_name_for_db}' already has {num_completed_trials} trials, as the limit  of the target was {TRIALS_PER_OPTIMIZER}.")
            print(f"Skipping new optimization for {opt_type}.")
        else:
            print(f"Study '{study_name_for_db}' has {num_completed_trials} trials completed. Running {remaining_trials} more trials to reach {TRIALS_PER_OPTIMIZER}.")
            study.optimize(
                create_objective_for_optimizer(opt_type), 
                n_trials=remaining_trials, 
                gc_after_trial=True, 
                callbacks=[print_progress_callback] 
            )
        
        # Always print best trial info after potentially running new trials
        if study.best_trial is not None:
            print(f"\nStudy finished for {opt_type}. Best trial:")
            print(f"  Value (Validation Loss): {study.best_value:.4f}")
            print(f"  Params: {study.best_params}")
            
            # Collect data for combined plotting
            all_best_trials_data[opt_type] = {
                'train_losses': study.best_trial.user_attrs.get("train_losses", []),
                'val_losses': study.best_trial.user_attrs.get("val_losses", []),
                'val_accuracies': study.best_trial.user_attrs.get("val_accuracies", []),
                # No final test metrics from Optuna trials, as that's a separate step
            }
        else:
            print(f"\nNo successful trials found for {opt_type} in study '{study_name_for_db}'.")

    print("\n-All Optimizer Studies Complete-")


    combined_plot_output_dir = os.path.join(STUDY_STORAGE_BASE, "plots")
    os.makedirs(combined_plot_output_dir, exist_ok=True) 

    plot_combined_optimizer_metrics(
        all_opt_data=all_best_trials_data, 
        plot_save_dir=combined_plot_output_dir
    )

    print("\nOptimization process complete. 'optuna_studies/plots' folder for comparison plots.")
    

