import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from models import *


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),  # Adding this line to convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.2]),  # Use single-channel values for grayscale
])

# Define paths to your datasets
train_dataset_path = './dataset/train'
test_dataset_path = './dataset/test'

# transform the data to get the clean data
train_dataset = datasets.ImageFolder(root=train_dataset_path, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dataset_path, transform=transform)

# Load and transform datasets
train_dataset = datasets.ImageFolder(root=train_dataset_path, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dataset_path, transform=transform)

# Split datasets into training and validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Define DataLoaders
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Initialize models
models = {
    'convnet7': ConvNet7(num_classes=5).to(device),
    'convnet7_deeper': ConvNet7_deeper(num_classes=5).to(device),
    'convnet3_deeper': ConvNet3_deeper(in_channels=1, out_channels=5).to(device)
}

# Define paths for model saving
save_dir = './saved_models/'
os.makedirs(save_dir, exist_ok=True)

# Define loss function, optimizers, and schedulers for each model
criterion = nn.CrossEntropyLoss()

optimizers = {
    'convnet7': optim.Adam(models['convnet7'].parameters(), lr=0.001),
    'convnet7_deeper': optim.Adam(models['convnet7_deeper'].parameters(), lr=0.001),
    'convnet3_deeper': optim.Adam(models['convnet3_deeper'].parameters(), lr=0.001)
}

schedulers = {
    'convnet7': StepLR(optimizers['convnet7'], step_size=30, gamma=0.1),
    'convnet7_deeper': StepLR(optimizers['convnet7_deeper'], step_size=30, gamma=0.1),
    'convnet3_deeper': StepLR(optimizers['convnet3_deeper'], step_size=30, gamma=0.1)
}

# Training function with Early Stopping
def train_and_save(model, optimizer, scheduler, criterion, epochs, train_dataloader, val_dataloader, device, save_path, patience=10):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(epochs), desc="Training"):
        # Training phase
        model.train()
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += criterion(outputs, labels).item()

        val_loss /= len(val_dataloader)
        val_accuracy = val_correct / val_total

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0  # reset counter
            print(f"Epoch {epoch+1}: Validation loss improved, model saved.")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}: No improvement in validation loss for {patience_counter} epochs.")

        if patience_counter >= patience:
            print(f"Stopping early after {epoch+1} epochs due to no improvement in validation loss.")
            break

    return model

# Train and save models with Early Stopping
patience_value = 10  # number of epochs to wait after last improvement

# Train and save models
for model_name in models:
    print(f"Training and saving {model_name}")
    train_and_save(models[model_name],
                   optimizers[model_name],
                   schedulers[model_name],
                   criterion,
                   epochs=60,  # Total epochs to train if early stopping doesn't trigger
                   train_dataloader=train_dataloader,
                   val_dataloader=val_dataloader,
                   device=device,
                   save_path=os.path.join(save_dir, f"{model_name}.pt"),
                   patience=patience_value)
