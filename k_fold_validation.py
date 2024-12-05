import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchinfo import summary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import pandas as pd
from torch.optim.lr_scheduler import StepLR
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from models import *

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

# Define the path to the saved models
model_paths = {
    'convnet7': './saved_models/convnet7.pt',
    'convnet7_deeper': './saved_models/convnet7_deeper.pt',
    'convnet3_deeper': './saved_models/convnet3_deeper.pt'
}

# Define the classes
classes = ['Angry', 'Focused', 'Happy', 'Netural', 'Surprised']

# Load Models
def load_model(model_path, model_class, device):
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Define models
models = {
    'convnet7': load_model(model_paths['convnet7'], ConvNet7, device),
    'convnet7_deeper': load_model(model_paths['convnet7_deeper'], ConvNet7_deeper, device),
    'convnet3_deeper': load_model(model_paths['convnet3_deeper'], ConvNet3_deeper, device)
}

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate Models
def evaluate_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    y_true = []
    y_pred = []

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.view(-1).tolist())
        y_true.extend(labels.view(-1).tolist())

    accuracy = accuracy_score(y_true, y_pred)
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    conf_mat = confusion_matrix(y_true, y_pred)

    return accuracy, precision_micro, precision_macro, recall_micro, recall_macro, f1_micro, f1_macro, conf_mat

# Define your models as a dictionary of classes, not instances
model_classes = {
    'convnet7': ConvNet7,
    'convnet3_deeper': ConvNet3_deeper,
    'convnet7_deeper': ConvNet7_deeper
}


def k_fold_cross_validation(model_class, epochs, dataset, device, saved_file_prefix, n_splits=10, batch_size=32, patience=10):
    print("starrted k_fold_cross_validation ...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    targets = [y for _, y in dataset]  # Ensure you have a list of targets if not directly accessible
    all_fold_results = []  # Store results for each fold

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), targets)):
        print(f"Fold {fold + 1}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)

        # Initialize the model, optimizer, and scheduler for this fold
        model = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        load_file =  f"{saved_file_prefix}.pt"
        save_path = f"{saved_file_prefix}_fold_{fold + 1}.pt"
        train_and_save(model, optimizer, scheduler, criterion, epochs, train_loader, val_loader, device, save_path, patience)

        # Load the best model for this fold
        model.load_state_dict(torch.load(save_path))

        # Evaluate the model on thec validation set
        p_name = saved_file_prefix.split("/")[-1]
        name = f"{p_name}_fold_{fold + 1}"
        accuracy, precision_micro, precision_macro, recall_micro, recall_macro, f1_micro, f1_macro, conf_mat = evaluate_model(model, val_loader, device)
        print(f"Metrics for {name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (Micro): {precision_micro:.4f}, Precision (Macro): {precision_macro:.4f}")
        print(f"Recall (Micro): {recall_micro:.4f}, Recall (Macro): {recall_macro:.4f}")
        print(f"F1 Score (Micro): {f1_micro:.4f}, F1 Score (Macro): {f1_macro:.4f}")
        print("Confusion Matrix:")
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for {name}')
        plt.show()
        # Store the results
        # all_fold_results.append(fold_results)

    # Compute average results across all folds
    # avg_results = np.mean(np.array([r.values() for r in all_fold_results]), axis=0)
    return None

full_dataset = datasets.ImageFolder(root=train_dataset_path, transform=transform)


for model_name, model_cls in model_classes.items():
    print(f"Training and saving k-fold for {model_name}")
    all_fold_results, avg_results = k_fold_cross_validation(
        model_class=model_cls,  # Pass class, ensuring a fresh instance is created each time
        epochs=60,
        dataset=full_dataset,
        device=device,
        saved_file_prefix=os.path.join(save_dir, f"k_fold_{model_name}"),
        n_splits=10,
        batch_size=32,
        patience=10
    )
    # Output the results for each fold and the average
    print(f"Fold results for {model_name}: {all_fold_results}")
    print(f"Average results for {model_name}: {avg_results}")