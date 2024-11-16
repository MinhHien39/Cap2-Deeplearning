import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
import time
import datetime
from torchvision import models, transforms
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class CustomCNN(nn.Module):
    def __init__(self, num_classes=14):
        super(CustomCNN, self).__init__()
        # CNN Layers
        self.conv_layers = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Second Convolutional Block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Third Convolutional Block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Fourth Convolutional Block with Dropout
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
        )
        
        # ResNet feature extractor
        self.resnet = models.resnet50(pretrained=True)
        # Freeze early layers
        for param in list(self.resnet.parameters())[:-10]:
            param.requires_grad = False
            
        # Combine features
        self.combine_features = nn.Sequential(
            nn.Linear(2048 + 512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # CNN path
        cnn_features = self.conv_layers(x)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        
        # ResNet path
        resnet_features = self.resnet(x)
        
        # Combine features
        combined = torch.cat((cnn_features, resnet_features), dim=1)
        return self.combine_features(combined)

class YOLOCNNModel:
    def __init__(self, data_yaml, weights='yolov8n.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Khởi tạo YOLO model
        self.yolo = YOLO(weights)
        
        # Khởi tạo CNN model
        self.cnn = CustomCNN().to(self.device)
        
        # Optimizer với learning rate scheduler
        self.optimizer = optim.AdamW(
            self.cnn.parameters(), 
            lr=0.001, 
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.1, 
            patience=5,
            verbose=True
        )
        
        # Loss function với class weights
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        self.data_yaml = data_yaml
        self.best_accuracy = 0
        self.early_stopping_counter = 0
        self.early_stopping_patience = 10
        
    def calculate_class_weights(self, labels):
        """Tính toán class weights để xử lý imbalanced data"""
        class_counts = torch.bincount(labels)
        total = len(labels)
        weights = total / (len(class_counts) * class_counts.float())
        return weights.to(self.device)
        
    def train_epoch(self, epoch, total_epochs, train_loader):
        epoch_start_time = time.time()
        
        # Train YOLO
        results = self.yolo.train(
            data=self.data_yaml,
            epochs=1,
            imgsz=64,
            batch=32,
            exist_ok=True,
            resume=True
        )
        
        # Train CNN
        self.cnn.train()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(self.device), targets.to(self.device)
            
            # Get YOLO features
            with torch.no_grad():
                yolo_features = self.yolo.predict(images, verbose=False)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.cnn(images)  # Using original images for CNN
            
            # Calculate loss with class weights
            class_weights = self.calculate_class_weights(targets)
            loss = self.criterion(outputs, targets)
            weighted_loss = (loss * class_weights[targets]).mean()
            
            # Backward pass
            weighted_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.cnn.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            running_loss += weighted_loss.item()
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            
            # Print batch progress
            if (batch_idx + 1) % 10 == 0:
                print(f'Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {weighted_loss.item():.4f}')
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, 
            all_predictions, 
            average='weighted'
        )
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch results
        print(f'\nEpoch [{epoch}/{total_epochs}]')
        print(f'Average Loss: {running_loss/len(train_loader):.4f}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')
        print(f'Epoch Time: {datetime.timedelta(seconds=int(epoch_time))}')
        
        return accuracy

    def validate(self, val_loader):
        """Đánh giá model trên validation set"""
        self.cnn.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.cnn(images)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, 
            all_predictions, 
            average='weighted'
        )
        
        print('\nValidation Results:')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')
        
        return accuracy

    def train(self, train_loader, val_loader, epochs=100, save_path='modelYolo2.pt'):
        print(f"Training on device: {self.device}")
        total_start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            print(f"\n{'='*50}")
            
            # Train epoch
            train_accuracy = self.train_epoch(epoch, epochs, train_loader)
            
            # Validate
            val_accuracy = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_accuracy)
            
            # Save best model
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                torch.save({
                    'epoch': epoch,
                    'yolo_state_dict': self.yolo.state_dict(),
                    'cnn_state_dict': self.cnn.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_accuracy': self.best_accuracy,
                }, save_path)
                print(f"Saved new best model with accuracy: {val_accuracy:.4f}")
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        total_time = time.time() - total_start_time
        print(f"\nTotal training time: {datetime.timedelta(seconds=int(total_time))}")
        print(f"Best validation accuracy: {self.best_accuracy:.4f}")

    def resume_training(self, checkpoint_path='modelYolo2.pt'):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.yolo.load_state_dict(checkpoint['yolo_state_dict'])
            self.cnn.load_state_dict(checkpoint['cnn_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_accuracy = checkpoint['best_accuracy']
            start_epoch = checkpoint['epoch']
            print(f"Resumed from epoch {start_epoch} with accuracy {self.best_accuracy:.4f}")
            return start_epoch
        return 0

def main():
    # Initialize data loaders
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load your data using custom DataLoader
    train_loader = DataLoader(
        YourDataset(root_dir="path/to/train", transform=transform),
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        YourDataset(root_dir="path/to/val", transform=transform),
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    data_yaml = "path/to/your/data.yaml"
    model = YOLOCNNModel(data_yaml)
    
    # Resume training if checkpoint exists
    start_epoch = model.resume_training()
    
    # Train model
    model.train(train_loader, val_loader, epochs=100-start_epoch)

if __name__ == "__main__":
    main()