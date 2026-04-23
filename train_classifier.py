"""
Binary Classifier for Tumor Detection
Trains on Brain Tumor MRI Dataset (Kaggle)
Classes: 0 = No Tumor, 1 = Tumor (glioma/meningioma/pituitary)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import random
import numpy as np
from sklearn.metrics import classification_report


# ─── CONFIG ───────────────────────────────────────────────────────
TRAINING_DIR = "Training"
TESTING_DIR = "Testing"
MODEL_DIR = "models"
BATCH_SIZE = 32
EPOCHS = 3
LR = 1e-4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─── DATASET ──────────────────────────────────────────────────────
class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.samples = []
        
        classes = {
            'notumor': 0,
            'glioma': 1,
            'meningioma': 1,
            'pituitary': 1
        }
        for class_name, label in classes.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.jpg'):
                        self.samples.append((os.path.join(class_dir, img_name), label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ─── MODEL ────────────────────────────────────────────────────────
def create_classifier():
    """Create ResNet18 classifier."""
    model = models.resnet18(pretrained=True)
    # Freeze the feature extractor for faster CPU training
    for param in model.parameters():
        param.requires_grad = False
    # Modify final layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


# ─── TRAINING ─────────────────────────────────────────────────────
def train_classifier():
    print("🧠 Training Binary Tumor Classifier")
    print(f"   Device: {DEVICE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print()
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Datasets
    train_dataset = BrainTumorDataset(TRAINING_DIR, transform=train_transform, is_train=True)
    val_dataset = BrainTumorDataset(TESTING_DIR, transform=val_transform, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # Model
    model = create_classifier()
    model = model.to(DEVICE)
    
    # Loss, Optimizer
    # Loss, Optimizer (Handle class imbalance: 1400 No Tumor vs 4200 Tumor -> weight 3:1)
    class_weights = torch.tensor([3.0, 1.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = 100. * correct / total
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print()
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "classifier.pth"))
            print("   ✅ Best model saved!")
    
    print(f"\n🎉 Training complete!")
    print(f"   Best Val Acc: {best_acc:.2f}%")    
    # Final classification report
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['No Tumor', 'Tumor']))


if __name__ == "__main__":
    train_classifier()