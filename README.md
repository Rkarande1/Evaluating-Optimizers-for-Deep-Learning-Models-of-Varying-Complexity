# A Comparative Study of Optimisers Across Tasks and Model Complexities

## Technical Report for Deep Learning Optimization Benchmarking

This repository contains the code and resources related to the technical study: **"A Comparative Study of Optimisers Across Tasks and Model Complexities - LeNet & U-net."** The report provides a comprehensive comparative analysis of common optimization algorithms in deep learning.

The project investigates the performance, convergence speed, and stability of five optimizers when training two distinct Convolutional Neural Network (CNN) architectures **from scratch**. This approach provides an unbiased view of each optimizer's viability across model complexity and task type.

### Keywords
Deep Learning, Convolutional Neural Networks, LeNet5, UNet, Image Classification, Semantic Segmentation, CIFAR-10, PASCAL VOC, Optimisers, Adam, AdamW, SGD, RMSprop, Training from Scratch.

---

## Key Findings & Results

The study tested the performance of **SGD**, **SGD+Momentum+WeightDecay**, **RMSprop**, **Adam**, and **AdamW** on two different tasks. The findings showed successful model convergence for both architectures across the different optimizers, presenting their unique characteristics.

| Architecture | Task | Best Optimizer | Test Metric | Result | Insight |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LeNet-5** (Classifier) | Image Classification (CIFAR-10) | **AdamW** | Test Accuracy | **70.08%** | AdamW converged the earliest in validation loss and achieved the highest final test accuracy. |
| **UNet** (Segmenter) | Semantic Segmentation (Pascal VOC 2012) | **Adam** | Test mIoU | **6.18%** | Adaptive learning rates helped navigate the complex loss landscape[cite: 155]. Adam consistently reached the highest mIoU. |

### Core Takeaways:

* **For Simple Tasks (LeNet-5/CIFAR-10):** Optimizers like **AdamW** and **Adam**, along with **SGD with Momentum**, consistently achieved test accuracies of roughly **$70\%$**[cite: 149].
* **For Complex Tasks (UNet/Pascal VOC):** **Adam** and **AdamW** pulled ahead of others with faster convergence [cite: 167] and better final performance[cite: 154], demonstrating that adaptive learning rates are highly beneficial.
* **The SGD+Momentum+Weight Decay** optimizer struggled significantly with the UNet segmentation task, achieving only **$1.28\%$ mIoU** [cite: 143, 158], suggesting difficulty adapting to a rugged loss landscape.

---

## Methodology

### 1. Model Architectures

* **LeNet-5:** Adapted for the CIFAR-10 image classification task. It uses two sequential blocks of convolutional and max-pooling layers, followed by three fully connected layers. **ReLU** activation was used instead of Tanh for simplification and faster training.
* **UNet:** Used for semantic segmentation. This architecture is known for its symmetric encoder-decoder structure and **skip connections** , providing pixel-wise predictions for 21 classes.

### 2. Datasets

| Dataset | Task | Characteristics |
| :--- | :--- | :--- |
| **CIFAR-10**  | Image Classification | 60,000 $32\times32$ color images in 10 classes; **perfectly balanced**]. |
| **Pascal VOC 2012**  | Semantic Segmentation | Images with pixel-level annotations for 20 foreground classes; has a **very high data imbalance**. |

### 3. Training and Evaluation

* **Hyperparameter Tuning:** **Optuna** was used to test different parameter combinations for all optimizers prior to the final runs.
* **LeNet-5 Evaluation:** Performance was judged using **Cross-Entropy Loss** and **Accuracy**.
* **UNet Evaluation:** Performance was judged using **Cross-Entropy Loss with Class Weights** (to mitigate imbalance) , **Mean Intersection over Union (mIoU)** , and **Pixel Accuracy**.

---

## Getting Started

### Prerequisites

* Python 3.x
* PyTorch (CUDA recommended for UNet)
* Git

### Installation

1.  **Clone the repository:**
    
    git clone https://github.com/Rkarande1/Evaluating-Optimizers-for-Deep-Learning-Models-of-Varying-Complexity.git
    cd your-repo-name
    

2.  **Install dependencies:**
    
    pip install -r requirements.txt
    

### Running the Experiments

The following scripts execute the final comparative training runs using the best-found hyperparameters:

# Training LeNet-5 (Classification)
python scripts/train_lenet.py

# Training UNet (Segmentation)
python scripts/train_unet.py

# References
The architecture definitions and core concepts are based on the following seminal works:

[1] LeNet-5: Y. LeCun et al. (1998). Gradient-based learning applied to document recognition.

[7] UNet: O. Ronneberger et al. (2015). U-Net: Convolutional networks for biomedical image segmentation.

[6] Adam: Kingma, D. P. & Ba, J. (2015). Adam: A method for stochastic optimization.

[9] AdamW: Loshchilov, I. & Hutter, F. (2019). Decoupled weight decay regularization.

[8] Optuna: Akiba, T. et al. (2019). Optuna: A next-generation hyperparameter optimization framework.
