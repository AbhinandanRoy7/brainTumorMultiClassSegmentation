"""
Quick Training Script for Brain Tumor MRI Segmentation
Uses a small subset for fast CPU training (~10-15 min)
"""

import os
import glob
import random
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import time


# ─── CONFIG ───
DATA_DIR = "data"
MODEL_DIR = "models"
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3
NUM_CLASSES = 4
IN_CHANNELS = 4
IMG_SIZE = 240
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
MAX_TRAIN_SAMPLES = 240   # Small subset for fast CPU training
MAX_VAL_SAMPLES = 80

os.makedirs(MODEL_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class BraTSDataset(Dataset):
    def __init__(self, file_paths, augment=False):
        self.file_paths = file_paths
        self.augment = augment
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        with h5py.File(path, 'r') as f:
            image = f['image'][()]
            mask = f['mask'][()]
        
        label = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int64)
        label[mask[:, :, 0] == 1] = 1
        label[mask[:, :, 1] == 1] = 2
        label[mask[:, :, 2] == 1] = 3
        
        image = image.transpose(2, 0, 1).astype(np.float32)
        
        if self.augment and random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
            label = np.flip(label, axis=1).copy()
        
        return torch.tensor(image), torch.tensor(label)


def dice_score(pred, target, num_classes=4, smooth=1e-6):
    dices = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        dice = (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
        dices.append(dice.item())
    return dices


def dice_loss(pred_logits, target, num_classes=4, smooth=1e-6):
    pred_probs = torch.softmax(pred_logits, dim=1)
    loss = 0.0
    for c in range(num_classes):
        pred_c = pred_probs[:, c]
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        loss += 1 - (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
    return loss / num_classes


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_weights = torch.tensor([0.05, 0.3, 0.3, 0.35]).to(DEVICE)
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)

        pred_probs = torch.softmax(pred, dim=1)

        dice_loss_val = 0.0
        smooth = 1e-5

        # Ignore background
        for c in range(1, pred.shape[1]):
            pred_c = pred_probs[:, c]
            target_c = (target == c).float()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice_loss_val += 1 - (2. * intersection + smooth) / (union + smooth)

        dice_loss_val /= (pred.shape[1] - 1)

        return 0.5 * ce_loss + 0.5 * dice_loss_val

# ─── TRAINING ───
def train():
    print(f"⚡ Quick Training Mode (CPU)")
    print(f"   Device: {DEVICE}")
    print(f"   Epochs: {EPOCHS}, Batch: {BATCH_SIZE}")
    print(f"   Max Train: {MAX_TRAIN_SAMPLES}, Max Val: {MAX_VAL_SAMPLES}")
    print()
    
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5")))
    valid_files = [f for f in all_files if os.path.getsize(f) > 20000]
    print(f"📂 Valid slices: {len(valid_files)}")
    
    random.shuffle(valid_files)
    
    # Small subset
    train_files = valid_files[:MAX_TRAIN_SAMPLES]
    val_files = valid_files[MAX_TRAIN_SAMPLES:MAX_TRAIN_SAMPLES + MAX_VAL_SAMPLES]
    print(f"   Train: {len(train_files)}, Val: {len(val_files)}")
    
    train_loader = DataLoader(BraTSDataset(train_files, augment=True), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(BraTSDataset(val_files), 
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Lighter encoder for faster CPU training
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
    )
    model = model.to(DEVICE)
    
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'wt_dice': [], 'tc_dice': [], 'et_dice': [],
        'best_val_loss': float('inf'), 'best_epoch': 0
    }
    
    print(f"\n🚀 Training started...\n")
    
    for epoch in range(EPOCHS):
        start = time.time()
        
        # Train
        model.train()
        train_loss = 0.0
        train_dices = [0.0] * NUM_CLASSES
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            batch_dices = dice_score(preds, labels)
            for c in range(NUM_CLASSES):
                train_dices[c] += batch_dices[c]
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  [{epoch+1}/{EPOCHS}] Batch {batch_idx+1}/{len(train_loader)} Loss: {loss.item():.4f}")
        
        num_train_batches = len(train_loader)
        train_loss /= num_train_batches
        train_dices = [d / num_train_batches for d in train_dices]
        mean_train_dice = np.mean(train_dices[1:])
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_dices = [0.0] * NUM_CLASSES
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                batch_dices = dice_score(preds, labels)
                for c in range(NUM_CLASSES):
                    val_dices[c] += batch_dices[c]
        
        num_val_batches = len(val_loader)
        val_loss /= num_val_batches
        val_dices = [d / num_val_batches for d in val_dices]
        mean_val_dice = np.mean(val_dices[1:])
        
        scheduler.step(val_loss)
        elapsed = time.time() - start
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(float(mean_train_dice))
        history['val_dice'].append(float(mean_val_dice))
        history['wt_dice'].append(val_dices[1])
        history['tc_dice'].append(val_dices[2])
        history['et_dice'].append(val_dices[3])
        
        print(f"\n📊 Epoch {epoch+1}/{EPOCHS} ({elapsed:.0f}s)")
        print(f"   Train Loss: {train_loss:.4f} | Dice: {mean_train_dice:.4f}")
        print(f"   Val   Loss: {val_loss:.4f} | Dice: {mean_val_dice:.4f}")
        print(f"   WT: {val_dices[1]:.4f} | TC: {val_dices[2]:.4f} | ET: {val_dices[3]:.4f}")
        
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch + 1
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
            print(f"   ✅ Best model saved!")
        print()
    
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "last_model.pth"))
    with open(os.path.join(MODEL_DIR, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"🎉 Training complete!")
    print(f"   Best epoch: {history['best_epoch']}")
    print(f"   Best val loss: {history['best_val_loss']:.4f}")


if __name__ == "__main__":
    train()
