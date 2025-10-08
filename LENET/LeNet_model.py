import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(LeNet5, self).__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels 

        
        self.features = nn.Sequential(
            # C1: Convolutional layer 1
            
            nn.Conv2d(in_channels, 6 * in_channels, kernel_size=5),
            nn.ReLU(), # Changed from nn.Tanh() 
            
            # S2: Max Pooling layer 1
            
            nn.MaxPool2d(kernel_size=2),
            
            # C3: Convolutional layer 2
        
            nn.Conv2d(6 * in_channels, 16 * in_channels, kernel_size=5),
            nn.ReLU(), # Changed from nn.Tanh()
            
            # S4: Max Pooling layer 2
            
            nn.MaxPool2d(kernel_size=2)
        )

        # Classifier layers (Fully Connected)
        
        self.classifier = nn.Sequential(
            # F5: Fully connected layer 1
           
            nn.Linear(16 * 5 * 5 * in_channels, 120 * in_channels),
            nn.ReLU(), # Changed from nn.Tanh() 
            
            # F6: Fully connected layer 2
            
            nn.Linear(120 * in_channels, 84 * in_channels),
            nn.ReLU(), # Changed from nn.Tanh() 
            
            # Output layer
         
            nn.Linear(84 * in_channels, num_classes),
        )

    def forward(self, x):
        
        x = self.features(x)
       
        x = torch.flatten(x, 1) 
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1) #  to get probabilities
        return logits, probas

