import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os

# Check if GPU is available
# if you are using macos
# device = torch.device("mps")
# if you are using windows
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Dataset directory
# data_dir = './images_biclassification/'
data_dir = 'E:/ANU/24s2/8715/DATA/images_biclassification/'

# Data augmentation and preprocessing
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load the full dataset (includes all data)
full_data = datasets.ImageFolder(root=data_dir, transform=data_transforms['train'])

# View class-to-index mapping
print("Class-to-index mapping:")
print(full_data.class_to_idx)  # Output example: {'implants': 0, 'no_implants': 1}

# Automatically split the dataset into train, val, and test sets
train_size = int(0.7 * len(full_data))  # 70% for training
val_size = int(0.15 * len(full_data))   # 15% for validation
test_size = len(full_data) - train_size - val_size  # 15% for testing
train_data, val_data, test_data = random_split(full_data, [train_size, val_size, test_size])

# Set different transforms for training, validation, and testing sets
train_data.dataset.transform = data_transforms['train']
val_data.dataset.transform = data_transforms['val_test']
test_data.dataset.transform = data_transforms['val_test']

# Data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)  # Set num_workers=0 for debugging
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)

# Calculate the number of samples for each class to set class weights
from collections import Counter
train_labels = [label for _, label in train_data]
label_counts = Counter(train_labels)
print(f"Training set label distribution: {label_counts}")

# Calculate weights: weights are inversely proportional to the number of samples
class_weights = torch.tensor([1.0 / label_counts[i] for i in range(len(full_data.classes))])
class_weights = class_weights.to(device)

# Check if the class weights match the number of classes
print(f"Class weights: {class_weights} (Expected length: {len(full_data.classes)})")

# Use a pre-trained model (e.g., ResNet18)
model = models.resnet18(pretrained=True)

# Modify the final fully connected layer for binary classification
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Move the model to the device (GPU or CPU)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model and save the best version
def train_model(model, criterion, optimizer, num_epochs=10):
    best_acc = 0.0  # Track the best validation accuracy
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

        # Validate the model
        val_acc = validate_model(model)
        
        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_classification_model.pth')  # Save the best model
            print(f'Best model saved (Validation Accuracy: {best_acc:.4f})')

# Validate the model
def validate_model(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    print(f'Validation Accuracy: {val_acc:.4f}')
    return val_acc

# Test the model on the test set
def test_model(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total
    print(f'Test Accuracy: {test_acc:.4f}')

# Run training
train_model(model, criterion, optimizer, num_epochs=10)

# Load the saved model and test
model.load_state_dict(torch.load('best_classification_model.pth'))
model = model.to(device)
test_model(model)

