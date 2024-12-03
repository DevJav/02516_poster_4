import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models
from datasets import FrameVideoDataset, FlowVideoDataset
from torchvision import transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

TRAIN = True
EPOCHS = 10

def evaluate(loader):
    correct, total = 0, 0
    with torch.no_grad():
        for frames, flows, labels in loader:
            frames, flows, labels = frames.to(device), flows.to(device), labels.to(device)
            outputs = model(frames, flows)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

import torch
import torch.nn as nn
from torchvision import models

import torch
import torch.nn as nn
import torchvision.models as models

class TwoStreamNetwork(nn.Module):
    def __init__(self, num_classes):
        super(TwoStreamNetwork, self).__init__()
        
        # Spatial stream (ResNet for RGB frames)
        self.spatial_stream = models.resnet18(pretrained=True)
        self.spatial_stream.conv1 = nn.Conv2d(
            in_channels=30,  # 10 frames × 3 channels (stacked RGB frames)
            out_channels=self.spatial_stream.conv1.out_channels,
            kernel_size=self.spatial_stream.conv1.kernel_size,
            stride=self.spatial_stream.conv1.stride,
            padding=self.spatial_stream.conv1.padding,
            bias=self.spatial_stream.conv1.bias,
        )
        self.spatial_stream.fc = nn.Linear(512, num_classes)  # Adjust for classification
        
        # Temporal stream (ResNet for optical flow)
        self.temporal_stream = models.resnet18(pretrained=True)
        self.temporal_stream.conv1 = nn.Conv2d(
            in_channels=18,  # 2 × (T - 1), e.g., 9 flow pairs for 10 frames
            out_channels=self.temporal_stream.conv1.out_channels,
            kernel_size=self.temporal_stream.conv1.kernel_size,
            stride=self.temporal_stream.conv1.stride,
            padding=self.temporal_stream.conv1.padding,
            bias=self.temporal_stream.conv1.bias,
        )
        self.temporal_stream.fc = nn.Linear(512, num_classes)
        
        # Combine both streams
        self.fc_combined = nn.Linear(num_classes * 2, num_classes)  # Combine predictions from both streams

    def forward(self, frames, flow):
        # Spatial stream: Process stacked RGB frames
        batch_size, channels, num_frames, height, width = frames.shape
        frames_stacked = frames.view(batch_size, channels * num_frames, height, width)  # Merge temporal dimension
        spatial_features = self.spatial_stream(frames_stacked)  # Shape: [batch_size, num_classes]
        
        # Temporal stream: Process optical flow stack
        temporal_features = self.temporal_stream(flow)  # Shape: [batch_size, num_classes]
        
        # Concatenate both streams
        combined_features = torch.cat((spatial_features, temporal_features), dim=1)  # Shape: [batch_size, num_classes * 2]
        
        # Final classification
        output = self.fc_combined(combined_features)  # Shape: [batch_size, num_classes]
        
        return output


from torch.utils.data import Dataset

class CombinedVideoDataset(Dataset):
    def __init__(self, frame_dataset, flow_dataset):
        assert len(frame_dataset) == len(flow_dataset), "Frame and flow datasets must have the same length"
        self.frame_dataset = frame_dataset
        self.flow_dataset = flow_dataset

    def __len__(self):
        return len(self.frame_dataset)

    def __getitem__(self, idx):
        frames, label = self.frame_dataset[idx]  # Assuming frames dataset returns (frame, label)
        flow, _ = self.flow_dataset[idx]  # Assuming flow dataset returns (flow, label)
        
        return frames, flow, label

if __name__ == '__main__':


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    root_dir = '/dtu/datasets1/02516/ucf101_noleakage'
    if not os.path.exists(root_dir):
        print('Dataset not found. Update root_dir.')
        exit()

    results_dir = 'flow_results'
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

    train_flow_dataset = FlowVideoDataset(root_dir=root_dir, split='train', resize=(64, 64))
    val_flow_dataset = FlowVideoDataset(root_dir=root_dir, split='val', resize=(64, 64))
    test_flow_dataset = FlowVideoDataset(root_dir=root_dir, split='test', resize=(64, 64))

    # Create combined datasets
    train_combined_dataset = CombinedVideoDataset(train_frameimage_dataset, train_flow_dataset)
    val_combined_dataset = CombinedVideoDataset(val_frameimage_dataset, val_flow_dataset)
    test_combined_dataset = CombinedVideoDataset(test_frameimage_dataset, test_flow_dataset)

    # Create combined dataloaders
    train_combined_loader = DataLoader(train_combined_dataset, batch_size=8, shuffle=True)
    val_combined_loader = DataLoader(val_combined_dataset, batch_size=8, shuffle=False)
    test_combined_loader = DataLoader(test_combined_dataset, batch_size=8, shuffle=False)


    classes = train_frameimage_dataset.df['label'].unique()

    model = TwoStreamNetwork(num_classes=len(classes))
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
            for frames, flow, labels in train_combined_loader:
                frames, flow, labels = frames.to(device), flow.to(device), labels.to(device)
                # print("Frames shape: ", frames.shape)
                # print("Flow shape: ", flow.shape)
                # print("Labels shape: ", labels.shape)
                # Frames shape:  torch.Size([8, 3, 10, 64, 64])                                                                                          
                # Flow shape:  torch.Size([8, 18, 64, 64])                                                                                               
                # Labels shape:  torch.Size([8]) 
                optimizer.zero_grad()
                outputs = model(frames, flow)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_combined_loader)
            train_losses.append(avg_loss)
            model.eval()
            val_acc = evaluate(val_combined_loader)
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
        with open('flow_results/epoch_avg_loss.txt', 'r') as f:
            for line in f:
                train_losses.append(float(line))

        with open('flow_results/epoch_val_acc.txt', 'r') as f:
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
    test_acc = evaluate(test_combined_loader)
    print(f'Final Test Accuracy: {test_acc:.2f}%')

    # Save test accuracy into a file
    with open(os.path.join(results_dir, 'test_accuracy.txt'), 'w') as f:
        f.write(f"{test_acc}\n")

    # Run prediction on one test video and visualize results
    print('Visualizing predictions for one test video')
    test_iter = iter(test_combined_loader)
    frames, flows, labels = next(test_iter)  # Get the first batch of test data
    frames, flows, labels = frames.to(device), flows.to(device), labels.to(device)
    
    # Get predictions for all frames in the batch
    with torch.no_grad():
        outputs = model(frames, flows)
        _, predicted = torch.max(outputs.data, 1)

    # Convert frames, flows, and predicted labels to CPU for visualization
    frames = frames.cpu()
    flows = flows.cpu()
    predicted_classes = [classes[p] for p in predicted]

    # Plot all 10 frames with their predicted labels
    plt.figure(figsize=(15, 12))
    for i in range(min(len(frames), 10)):  # Plot up to 10 frames
        frame = frames[i].permute(1, 2, 0).numpy()  # Rearrange dimensions for visualization
        flow = flows[i].permute(1, 2, 0).numpy()  # Rearrange dimensions for visualization
        plt.subplot(4, 5, i + 1)
        plt.imshow(frame)
        plt.title(predicted_classes[i])
        plt.axis('off')
        plt.subplot(4, 5, i + 11)
        plt.imshow(flow)
        plt.title(predicted_classes[i])
        plt.axis('off')
    
    plt.suptitle(f'Predictions for Test Video (True Label: {classes[labels[0].item()]})')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(results_dir, 'test_video_predictions.png'))
