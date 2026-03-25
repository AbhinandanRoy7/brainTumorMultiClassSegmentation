"""
Brain Tumor MRI Segmentation - Training Script
U-Net with ResNet50 encoder for 4-class segmentation
Dataset: BraTS2020 pre-sliced H5 files
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


# ─── CONFIG ───────────────────────────────────────────────────────
DATA_DIR = "data"
MODEL_DIR = "models"
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
NUM_CLASSES = 4  # Background + 3 tumor regions
IN_CHANNELS = 4  # T1, T1ce, T2, FLAIR
IMG_SIZE = 240
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)


# ─── SEED ─────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# ─── DATASET ──────────────────────────────────────────────────────
class BraTSDataset(Dataset):
    """
    Loads pre-sliced H5 files.
    Each H5 has:
      - 'image': (240, 240, 4) float64 — normalized modalities
      - 'mask':  (240, 240, 3) uint8   — one-hot (Edema, TC, ET)
    
    We convert mask to class indices (0-3):
      0 = Background
      1 = Edema (WT)
      2 = Tumor Core
      3 = Enhancing Tumor
    """
    def __init__(self, file_paths, augment=False):
        self.file_paths = file_paths
        self.augment = augment
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        
        with h5py.File(path, 'r') as f:
            image = f['image'][()]   # (240, 240, 4)
            mask = f['mask'][()]     # (240, 240, 3)
        
        # Convert one-hot mask to class indices
        # Priority: ET (class 3) > TC (class 2) > Edema (class 1)
        label = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int64)
        label[mask[:, :, 0] == 1] = 1  # Edema
        label[mask[:, :, 1] == 1] = 2  # Tumor Core
        label[mask[:, :, 2] == 1] = 3  # Enhancing Tumor
        
        # Transpose image: (H, W, C) -> (C, H, W)
        image = image.transpose(2, 0, 1).astype(np.float32)
        
        # Simple augmentation
        if self.augment:
            if random.random() > 0.5:
                image = np.flip(image, axis=2).copy()
                label = np.flip(label, axis=1).copy()
            if random.random() > 0.5:
                image = np.flip(image, axis=1).copy()
                label = np.flip(label, axis=0).copy()
            if random.random() > 0.5:
                image = np.rot90(image, k=1, axes=(1,2)).copy()
                label = np.rot90(label, k=1).copy()
        
        return torch.tensor(image), torch.tensor(label)


# ─── METRICS ──────────────────────────────────────────────────────
def dice_score(pred, target, num_classes=4, smooth=1e-6):
    """Compute per-class Dice score."""
    dices = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        dice = (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
        dices.append(dice.item())
    return dices


def dice_loss(pred_logits, target, num_classes=4, smooth=1e-6):
    """Differentiable Dice loss."""
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


# ─── TRAINING ─────────────────────────────────────────────────────
def train():
    print(f"🧠 Brain Tumor Segmentation Training")
    print(f"   Device: {DEVICE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Learning Rate: {LR}")
    print()
    
    # Gather all H5 files
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5")))
    print(f"📂 Total slices found: {len(all_files)}")
    
    # Filter out empty slices (files < 15KB likely have no brain content)
    valid_files = [f for f in all_files if os.path.getsize(f) > 15000]
    print(f"📂 Valid slices (with content): {len(valid_files)}")
    
    # Shuffle and split: 80% train, 20% val
    random.shuffle(valid_files)
    split = int(0.8 * len(valid_files))
    train_files = valid_files[:split]
    val_files = valid_files[split:]
    
    print(f"   Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Datasets & Loaders
    train_dataset = BraTSDataset(train_files, augment=True)
    val_dataset = BraTSDataset(val_files, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    
    # Model: U-Net with ResNet50 encoder
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
    )
    model = model.to(DEVICE)
    
    # Loss, Optimizer, Scheduler
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'wt_dice': [], 'tc_dice': [], 'et_dice': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0
    }
    
    print(f"\n🚀 Training started...\n")
    
    for epoch in range(EPOCHS):
        start = time.time()
        
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_dices = [0.0] * NUM_CLASSES
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
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
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        train_dices = [d / len(train_loader) for d in train_dices]
        mean_train_dice = np.mean(train_dices[1:])  # Exclude background
        
        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_dices = [0.0] * NUM_CLASSES
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                batch_dices = dice_score(preds, labels)
                for c in range(NUM_CLASSES):
                    val_dices[c] += batch_dices[c]
        
        val_loss /= len(val_loader)
        val_dices = [d / len(val_loader) for d in val_dices]
        mean_val_dice = np.mean(val_dices[1:])
        
        scheduler.step(val_loss)
        
        elapsed = time.time() - start
        
        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(mean_train_dice)
        history['val_dice'].append(mean_val_dice)
        history['wt_dice'].append(val_dices[1])
        history['tc_dice'].append(val_dices[2])
        history['et_dice'].append(val_dices[3])
        
        print(f"\n📊 Epoch {epoch+1}/{EPOCHS} ({elapsed:.1f}s)")
        print(f"   Train Loss: {train_loss:.4f} | Train Dice: {mean_train_dice:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | Val Dice:   {mean_val_dice:.4f}")
        print(f"   WT: {val_dices[1]:.4f} | TC: {val_dices[2]:.4f} | ET: {val_dices[3]:.4f}")
        
        # Save best
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch + 1
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
            print(f"   ✅ Best model saved! (Val Loss: {val_loss:.4f})")
        
        print()
    
    # Save last model and history
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "last_model.pth"))
    
    with open(os.path.join(MODEL_DIR, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n🎉 Training complete!")
    print(f"   Best epoch: {history['best_epoch']}")
    print(f"   Best val loss: {history['best_val_loss']:.4f}")
    print(f"   Final WT Dice: {history['wt_dice'][-1]:.4f}")
    print(f"   Final TC Dice: {history['tc_dice'][-1]:.4f}")
    print(f"   Final ET Dice: {history['et_dice'][-1]:.4f}")


if __name__ == "__main__":
    train()
