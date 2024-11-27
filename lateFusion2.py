import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models
from datasets import FrameImageDataset
from torchvision import transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

TRAIN = True
EPOCHS = 50

def evaluate(loader):
    correct, total = 0, 0
    with torch.no_grad():
        for frames, labels in loader:
            video_class = labels[0]
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)

            # Aggregate logits directly (late fusion)
            avg_logits = torch.mean(outputs, dim=0)  # Average logits across frames

            # Apply softmax to aggregated logits
            avg_probabilities = nn.functional.softmax(avg_logits, dim=0)

            # Determine the predicted class
            predicted_class = torch.argmax(avg_probabilities).item()
            total += 1
            if video_class == predicted_class:
                correct += 1
    return 100 * correct / total


def visualize_predictions(test_loader, batch_index=0):
    print(f'Visualizing predictions for test video batch {batch_index}')
    test_iter = iter(test_loader)
    
    # Skip to the desired batch
    for _ in range(batch_index + 1):
        frames, labels = next(test_iter)
    
    frames, labels = frames.to(device), labels.to(device)
    
    # Get predictions for all frames in the batch
    with torch.no_grad():
        outputs = model(frames)

        # Aggregate logits directly (late fusion)
        avg_logits = torch.mean(outputs, dim=0)

        # Apply softmax to aggregated logits
        avg_probabilities = nn.functional.softmax(avg_logits, dim=0)

        # Determine the predicted class
        predicted_class = torch.argmax(avg_probabilities).item()

    # Convert frames and predicted labels to CPU for visualization
    frames = frames.cpu()
    predicted_class_name = classes[predicted_class]

    # Plot all 10 frames with their predicted labels
    plt.figure(figsize=(15, 6))
    for i in range(min(len(frames), 10)):  # Plot up to 10 frames
        frame = frames[i].permute(1, 2, 0).numpy()  # Rearrange dimensions for visualization
        plt.subplot(2, 5, i + 1)
        plt.imshow(frame)
        plt.title(predicted_class_name)
        plt.axis('off')
    
    plt.suptitle(f'Predictions for Test Video (True Label: {classes[labels[0].item()]})')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(results_dir, f'test_video_predictions_batch_{batch_index}.png'))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    root_dir = '/zhome/99/e/203497/MyProject/poster4/ufc10'
    if not os.path.exists(root_dir):
        print('Dataset not found. Update root_dir.')
        exit()

    results_dir = 'lateFus2_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    time_prefix = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    results_dir = os.path.join(results_dir, time_prefix)
    os.makedirs(results_dir)

    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])

    print('Loading datasets')
    train_frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='train', transform=transform)
    val_frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
    test_frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='test', transform=transform)

    train_frameimage_loader = DataLoader(train_frameimage_dataset, batch_size=8, shuffle=True)
    val_frameimage_loader = DataLoader(val_frameimage_dataset, batch_size=8, shuffle=False)
    test_frameimage_loader = DataLoader(test_frameimage_dataset, batch_size=8, shuffle=False)

    classes = train_frameimage_dataset.df['label'].unique()

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, len(classes))
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
        with open('lateFus2_results/epoch_avg_loss.txt', 'r') as f:
            for line in f:
                train_losses.append(float(line))

        with open('lateFus2_results/epoch_val_acc.txt', 'r') as f:
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
    visualize_predictions(test_frameimage_loader, batch_index=1)