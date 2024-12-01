import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models
from datasets import FrameVideoDataset
from torchvision import transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

TRAIN = True
EPOCHS = 10

def evaluate(loader):
    correct, total = 0, 0
    with torch.no_grad():
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

class EarlyFusionNN(nn.Module):
    def __init__(self, num_classes):
        super(EarlyFusionNN, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # Modify the first convolutional layer to accept 30 channels
        self.resnet.conv1 = nn.Conv2d(
            in_channels=30,  # Change from 3 to 30
            out_channels=self.resnet.conv1.out_channels,
            kernel_size=self.resnet.conv1.kernel_size,
            stride=self.resnet.conv1.stride,
            padding=self.resnet.conv1.padding,
            bias=self.resnet.conv1.bias
        )
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)	
        print(x.size())
        x = x.contiguous()
        print(x.size())

        # reshape images into T*3xHxW
        x = x.view(x.size(0), -1, x.size(3), x.size(4))
        print(x.size())

        return self.resnet(x)


if __name__ == '__main__':


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    root_dir = '/zhome/97/a/203937/02516_poster_4/ufc10'
    if not os.path.exists(root_dir):
        print('Dataset not found. Update root_dir.')
        exit()

    results_dir = 'earlyFusion_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    time_prefix = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    results_dir = os.path.join(results_dir, time_prefix)
    os.makedirs(results_dir)

    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])

    print('Loading datasets')
    train_frameimage_dataset = FrameVideoDataset(root_dir=root_dir, split='train', transform=transform)
    val_frameimage_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform)
    test_frameimage_dataset = FrameVideoDataset(root_dir=root_dir, split='test', transform=transform)

    train_frameimage_loader = DataLoader(train_frameimage_dataset, batch_size=8, shuffle=True)
    val_frameimage_loader = DataLoader(val_frameimage_dataset, batch_size=8, shuffle=False)
    test_frameimage_loader = DataLoader(test_frameimage_dataset, batch_size=8, shuffle=False)

    classes = train_frameimage_dataset.df['label'].unique()

    model = EarlyFusionNN(num_classes=len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_accuracies = []

    if TRAIN:
        print('Start Training')
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for frames, labels in train_frameimage_loader:
                frames, labels = frames.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(frames)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_frameimage_loader)
            train_losses.append(avg_loss)
            model.eval()
            val_acc = evaluate(val_frameimage_loader)
            val_accuracies.append(val_acc)
            print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}, Val Accuracy: {val_acc:.2f}%')
        print('Finished Training')

        # Save model
        torch.save(model.state_dict(), os.path.join(results_dir, 'model.pth'))

        # Save epoch average loss and validation accuracy into files
        with open(os.path.join(results_dir, 'epoch_avg_loss.txt'), 'w') as f:
            for loss in train_losses:
                f.write(f"{loss}\n")

        with open(os.path.join(results_dir, 'epoch_val_acc.txt'), 'w') as f:
            for acc in val_accuracies:
                f.write(f"{acc}\n")

    else:
        model.load_state_dict(torch.load('model.pth'))
        # Load epoch average loss and validation accuracy from files
        # to plot training results
        train_losses = []
        val_accuracies = []
        with open('earlyFusion_results/epoch_avg_loss.txt', 'r') as f:
            for line in f:
                train_losses.append(float(line))

        with open('earlyFusion_results/epoch_val_acc.txt', 'r') as f:
            for line in f:
                val_accuracies.append(float(line)) 

    plt.figure()
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss')
    plt.plot(range(1, EPOCHS + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Training Loss and Validation Accuracy')
    plt.savefig(os.path.join(results_dir, 'training_results.png'))

    model.eval()
    test_acc = evaluate(test_frameimage_loader)
    print(f'Final Test Accuracy: {test_acc:.2f}%')

    # Save test accuracy into a file
    with open(os.path.join(results_dir, 'test_accuracy.txt'), 'w') as f:
        f.write(f"{test_acc}\n")

    # Run prediction on one test video and visualize results
    print('Visualizing predictions for one test video')
    test_iter = iter(test_frameimage_loader)
    frames, labels = next(test_iter)  # Get the first batch of test data
    frames, labels = frames.to(device), labels.to(device)
    
    # Get predictions for all frames in the batch
    with torch.no_grad():
        outputs = model(frames)
        _, predicted = torch.max(outputs.data, 1)

    # Convert frames and predicted labels to CPU for visualization
    frames = frames.cpu()
    predicted_classes = [classes[p] for p in predicted]

    # Plot all 10 frames with their predicted labels
    plt.figure(figsize=(15, 6))
    for i in range(min(len(frames), 10)):  # Plot up to 10 frames
        frame = frames[i].permute(1, 2, 0).numpy()  # Rearrange dimensions for visualization
        plt.subplot(2, 5, i + 1)
        plt.imshow(frame)
        plt.title(predicted_classes[i])
        plt.axis('off')
    
    plt.suptitle(f'Predictions for Test Video (True Label: {classes[labels[0].item()]})')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(results_dir, 'test_video_predictions.png'))
