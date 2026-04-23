"""
Classifier for tumor detection.
Loads trained binary classifier.
"""

import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import numpy as np


class TumorClassifier:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _load_model(self, model_path):
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        model.eval()
        model.to(self.device)
        return model
    
    def predict(self, image):
        """
        Predict if image has tumor.
        image: PIL Image or numpy array (H, W, 3)
        Returns: 0 (no tumor) or 1 (tumor), confidence
        """
        if isinstance(image, np.ndarray):
            image = image.astype(np.float32)
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)
            elif image.ndim == 3 and image.shape[2] == 1:
                image = np.concatenate([image, image, image], axis=2)

            # Normalize float image range to [0, 255]
            if image.dtype != np.uint8:
                if image.max() > 1.0 or image.min() < 0.0:
                    if image.max() > image.min():
                        image = (image - image.min()) / (image.max() - image.min())
                    else:
                        image = np.clip(image, 0.0, 1.0)
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        return predicted.item(), confidence.item()