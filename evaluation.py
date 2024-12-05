import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from models import *

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Define the transformation and load the test dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.2])
])

test_dataset = datasets.ImageFolder(root='./dataset/test', transform=transform)
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


# Printing the metrics
for name, model in models.items():
    accuracy, precision_micro, precision_macro, recall_micro, recall_macro, f1_micro, f1_macro, conf_mat = evaluate_model(
        model, test_loader, device)
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