import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models
from datasets import *
from torchvision import transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class LateFusionModel(nn.Module):
    def __init__(self, num_classes):
        super(LateFusionModel, self).__init__()
        # Use a pre-trained 2D CNN (ResNet18)
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the final classification layer
        
        # Feature dimension from ResNet18
        self.feature_dim = 512
        
        # MLP for classification
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim * 10, 256),  # Aggregate features for 10 frames
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, frames):
        # frames shape: [batch_size, C, T, H, W]
        # print("################")
        # print(frames.size())  # Debugging

        # Permute to [batch_size, T, C, H, W]
        frames = frames.permute(0, 2, 1, 3, 4)  # Move T to the second dimension

        # Ensure the tensor is contiguous before reshaping
        frames = frames.contiguous()

        # Updated shape: [batch_size, T, C, H, W]
        batch_size, T, C, H, W = frames.size()
        
        # Reshape to process each frame independently
        frames = frames.view(batch_size * T, C, H, W)  # Shape: [batch_size * T, C, H, W]
        
        # Extract features for each frame using ResNet
        features = self.resnet(frames)  # Shape: (batch_size * T, feature_dim)
        
        # Reshape to aggregate frame features
        features = features.view(batch_size, T, -1)  # Shape: (batch_size, T, feature_dim)
        
        # Flatten all frames' features for each video
        features = features.view(batch_size, -1)  # Shape: (batch_size, T * feature_dim)
        
        # Pass through MLP for classification
        output = self.mlp(features) # Shape: [batch_size, num_classes]
        return output

class LateFusionRNNModel(nn.Module):
    def __init__(self, num_classes):
        super(LateFusionRNNModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove final layer
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),  # Dropout before the fully connected layer, to prevent OF
            nn.Linear(256, num_classes)
        )

    def forward(self, frames):
        # frames shape: [batch_size, T, C, H, W]
        frames = frames.permute(0, 2, 1, 3, 4)  # Shape: [batch_size, T, C, H, W]
        frames = frames.contiguous()
        batch_size, T, C, H, W = frames.size()

        # Reshape to process each frame independently
        frames = frames.view(batch_size * T, C, H, W)  # Shape: [batch_size * T, C, H, W]
        features = self.resnet(frames)  # Shape: [batch_size * T, 512]

        # Reshape back to [batch_size, T, 512] for LSTM
        features = features.view(batch_size, T, -1)
        lstm_out, _ = self.lstm(features)  # Shape: [batch_size, T, 256]
        output = self.fc(lstm_out[:, -1, :])  # Use the last hidden state for classification
        return output

def evaluate(loader, model, device, classes):
    correct, total = 0, 0
    with torch.no_grad():
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    root_dir = '/zhome/99/e/203497/MyProject/poster4/ufc10'

    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])

    # Use FrameVideoDataset with stacked frames
    train_dataset = FrameVideoDataset(root_dir=root_dir, split='train', transform=transform, stack_frames=True)
    val_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=True)
    test_dataset = FrameVideoDataset(root_dir=root_dir, split='test', transform=transform, stack_frames=True)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    classes = train_dataset.df['label'].unique()

    # Initialize Late Fusion model
    model = LateFusionRNNModel(num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    TRAIN = True
    EPOCHS = 50
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    if TRAIN:
        print('Start Training')
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            correct, total = 0, 0
            for frames, labels in train_loader:
                frames, labels = frames.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(frames)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Calculate training accuracy for the current batch
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_loss = running_loss / len(train_loader)
            train_losses.append(avg_loss)

            # Calculate and store training accuracy for the epoch
            train_acc = 100 * correct / total
            train_accuracies.append(train_acc)

            model.eval()
            val_acc = evaluate(val_loader, model, device, classes)
            val_accuracies.append(val_acc)
            print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}%, Train Accuracy: {train_acc} Val Accuracy: {val_acc:.2f}%')

        print('Finished Training')
        torch.save(model.state_dict(), 'late_fusion3_model.pth')

    else:
        model.load_state_dict(torch.load('late_fusion3_model.pth'))
        # Load epoch average loss and validation accuracy from files
        # to plot training results
        train_losses = []
        val_accuracies = []
        with open('aggregation_results/epoch_avg_loss.txt', 'r') as f:
            for line in f:
                train_losses.append(float(line))

        with open('aggregation_results/epoch_val_acc.txt', 'r') as f:
            for line in f:
                val_accuracies.append(float(line)) 

    results_dir = 'fusion3_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    time_prefix = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    results_dir = os.path.join(results_dir, time_prefix)
    os.makedirs(results_dir)

    plt.figure()
    # plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss')
    plt.plot(range(1, EPOCHS + 1), val_accuracies, label='Validation Accuracy')
    plt.plot(range(1, EPOCHS + 1), train_accuracies, label='Training Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig(os.path.join(results_dir, 'training_results.png'))
    
    # Evaluate on test set
    model.eval()
    test_acc = evaluate(test_loader, model, device, classes)
    print(f'Final Test Accuracy: {test_acc:.2f}%')
    
    # Save test accuracy into a file
    with open(os.path.join(results_dir, 'test_accuracy.txt'), 'w') as f:
        f.write(f"{test_acc}\n")
