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
from torchvision.models.video import r3d_18, R3D_18_Weights


class CNN3DModel(nn.Module):
    # Define the 3D CNN model, we don't use it, we are using a pre defined model insetad r3d_18
    def __init__(self, num_classes):
        super(CNN3DModel, self).__init__()
        
        # 3D Conv Layer 1: Input -> 3D Conv (3x3x3) -> Output Channels: 12
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=12, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.relu1 = nn.ReLU()
        
        # 3D Pool Layer 1: Pooling across spatial and temporal dimensions
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 4, 4), stride=(2, 4, 4))  # Pooling across temporal/spatial dimensions
        
        # 3D Conv Layer 2: Input -> 3D Conv (3x3x3) -> Output Channels: 24
        self.conv2 = nn.Conv3d(in_channels=12, out_channels=24, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
        # 3D Pool Layer 2
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # Further spatial/temporal reduction
        
        # Global Average Pooling: Reduce to (batch_size, 24)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Fully connected layer for classification
        self.fc = nn.Linear(24, num_classes)

    def forward(self, x):
        # x shape: [batch_size, C, T, H, W]
        print(x.size())
        x = self.conv1(x)  # Apply 3D convolution
        x = self.relu1(x)  # Apply ReLU activation
        print(x.size())
        x = self.pool1(x)  # Apply 3D pooling
        print(x.size())
        
        x = self.conv2(x)  # Apply second 3D convolution
        print(x.size())
        x = self.relu2(x)  # Apply ReLU activation
        # x = self.pool2(x)  # Apply 3D pooling
        
        x = self.global_pool(x)  # Global average pooling
        print(x.size())
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        print(x.size())
        x = self.fc(x)  # Classification layer
        return x

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
    results_dir = '3dcnn_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    time_prefix = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    results_dir = os.path.join(results_dir, time_prefix)
    os.makedirs(results_dir)
    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])

    # Use FrameVideoDataset with stacked frames
    train_dataset = FrameVideoDataset(root_dir=root_dir, split='train', transform=transform, stack_frames=True)
    val_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=True)
    test_dataset = FrameVideoDataset(root_dir=root_dir, split='test', transform=transform, stack_frames=True)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    classes = train_dataset.df['label'].unique()

    # Load pre-trained 3D ResNet18 model
    model = r3d_18(pretrained=True, weights=R3D_18_Weights.DEFAULT)

    # Modify the fully connected layer for your dataset's number of classes
    num_classes = len(classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Move to device
    model = model.to(device)
    # Initialize de model
    # model = CNN3DModel(num_classes=len(classes)).to(device)


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
            print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}%, Train Accuracy: {train_acc}% Val Accuracy: {val_acc:.2f}%')

        print('Finished Training')
        
        # Save model
        torch.save(model.state_dict(), os.path.join(results_dir, '3d_cnn_model.pth'))

        # Save epoch average loss and validation accuracy into files
        with open(os.path.join(results_dir, 'epoch_avg_loss.txt'), 'w') as f:
            for loss in train_losses:
                f.write(f"{loss}\n")

        with open(os.path.join(results_dir, 'epoch_val_acc.txt'), 'w') as f:
            for acc in val_accuracies:
                f.write(f"{acc}\n")
                
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
