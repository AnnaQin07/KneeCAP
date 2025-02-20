#############################################
# Classification: 0 - no implants
#                 1 - implants(hip)
#                 2 - implants(knee)
#                 3 - implants(ankle)
#                 4 - semi implants(knee)
#############################################
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from collections import Counter
import os
from PIL import Image

class AugmentedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, augmentation_config=None):
        super(AugmentedImageFolder, self).__init__(root, transform)
        self.augmentation_config = augmentation_config
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        # Apply different transforms based on class
        if self.augmentation_config and target in self.augmentation_config:
            # If the class is in the augmentation config, apply the augmentation transform
            sample = self.augmentation_config[target](sample)
        else:
            sample = self.transform(sample)
        
        return sample, target

class ImageClassificationModel:
    def __init__(self, data_dir, batch_size=32, num_epochs=10, learning_rate=0.001, device=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Set device: MPS, CUDA, or CPU
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS device")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA device")
        else:
            self.device = torch.device("cpu")
            print("Using CPU device")
        
        # Initialize the data transforms
        self.data_transforms = {
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

        # Define additional augmentations for minority classes
        self.augmentation_config = {
            3: transforms.Compose([  # Class 3 with only 3 images
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.5, contrast=0.5),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            4: transforms.Compose([  # Class 4 with only 3 images
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.5, contrast=0.5),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            1: transforms.Compose([  # Class 1 with 50+ images
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            2: transforms.Compose([  # Class 2 with 50+ images
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        if data_dir is not None:
            # Load dataset and split
            self._prepare_data()

            # Define the model
            self.model = self._build_model()
            self.model = self.model.to(self.device)

            # Define the loss function and optimizer
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        else:
            self.model = self._build_model()
            self.model = self.model.to(self.device)
    
    def _prepare_data(self):
        # Load the full dataset using the custom AugmentedImageFolder
        full_data = AugmentedImageFolder(
            root=self.data_dir,
            transform=self.data_transforms['train'],
            augmentation_config=self.augmentation_config  # Use the custom augmentation
        )
        
        # View class-to-index mapping
        print("Class-to-index mapping:")
        print(full_data.class_to_idx)
        
        # Automatically split the dataset into train, val, and test sets
        train_size = int(0.7 * len(full_data))
        val_size = int(0.15 * len(full_data))
        test_size = len(full_data) - train_size - val_size
        self.train_data, self.val_data, self.test_data = random_split(full_data, [train_size, val_size, test_size])
        
        # Set different transforms for validation and testing sets
        self.val_data.dataset.transform = self.data_transforms['val_test']
        self.test_data.dataset.transform = self.data_transforms['val_test']

        # Create data loaders
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=0)

        # Calculate class weights for imbalanced dataset
        train_labels = [label for _, label in self.train_data]
        label_counts = Counter(train_labels)
        print(f"Training set label distribution: {label_counts}")
        
        self.class_weights = torch.tensor([1.0 / label_counts[i] for i in range(len(full_data.classes))])
        self.class_weights = self.class_weights.to(self.device)
    
    def _build_model(self):
        # Use a pre-trained ResNet18 model and modify for 5-class classification
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5)  # Change to 5 output classes
        return model

    def train(self):
        best_acc = 0.0  # Track the best validation accuracy
        
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = correct / total
            print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

            # Validate the model
            val_acc = self.validate()
            
            # Save the best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_classification_model.pth')
                print(f'Best model saved (Validation Accuracy: {best_acc:.4f})')

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(f'Validation Accuracy: {val_acc:.4f}')
        return val_acc

    def test(self):
        self.model.load_state_dict(torch.load('best_classification_model.pth'))
        self.model = self.model.to(self.device)
        
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        test_acc = correct / total
        print(f'Test Accuracy: {test_acc:.4f}')
    
    def predict(self, image):
        """
        Given the path to an image, preprocess the image, pass it through the model,
        and output the predicted class label.
        """
        # Define the preprocessing steps
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3, -1, -1) if x.shape[0] == 1 else x),  # Convert to 3-channel if grayscale
            transforms.Lambda(lambda x: x.float()),  # Ensure the tensor is float type
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Apply preprocessing to the image
        image = preprocess(image).unsqueeze(0)  # Add batch dimension
        image = image.to(self.device)  # Move image to the correct device

        # Set the model to evaluation mode
        self.model.eval()

        # Disable gradient computation for inference
        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)  # Get the class with the highest score

        # Convert predicted index to class label
        class_index = predicted.item()  # Get the predicted class index
        # class_name = {v: k for k, v in self.train_data.dataset.class_to_idx.items()}[class_index]  # Map index to class name

        print(f'Predicted class for the input image: {class_index}')
        return class_index
    
    def load(self, best_model_path):
        self.model.load_state_dict(torch.load(best_model_path), strict=False)
        return self


# Example usage
if __name__ == '__main__':
    # data_dir = './images_biclassification/'
    # data_dir = '/Users/xinyao/Desktop/8715/images_biclassification/'
    data_dir =  'E:/ANU/24s2/8715/DATA/images_multiclassification/'
    model = ImageClassificationModel(data_dir=data_dir, num_epochs=10)
    model.train()
    model.test()

    # Predict the class of a new image
    image_path = 'E:/ANU/24s2/8715/DATA/images_multiclassification/2_implants/02236003_r.png'
    # Load and preprocess the image
    image = Image.open(image_path)
    prediction = model.predict(image)
    print(f'Predicted class for the input image: {prediction}')
