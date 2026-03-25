"""
Model definition and loading utilities.
Supports both ResNet34 (quick training) and ResNet50 (full training).
"""

import torch
import segmentation_models_pytorch as smp

NUM_CLASSES = 4
IN_CHANNELS = 4


def create_model(encoder_name="resnet34"):
    """Create U-Net model."""
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
    )
    return model


def load_model(model_path, device="cpu"):
    """Load trained model, auto-detecting encoder from checkpoint."""
    # Try ResNet34 first (quick training), then ResNet50
    for encoder in ["resnet34", "resnet50"]:
        try:
            model = create_model(encoder_name=encoder)
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)
            print(f"  Loaded model with {encoder} encoder")
            return model
        except RuntimeError:
            continue
    
    raise RuntimeError(f"Could not load model from {model_path}")
