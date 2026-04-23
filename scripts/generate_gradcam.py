import os
import sys
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.model import load_model, create_model
from backend.pipeline import load_h5_slice, find_tumor_slices, preprocess_for_inference, predict

def generate_gradcams():
    MODEL_PATH = "models/best_model.pth"
    DATA_DIR = "data"
    OUTPUT_DIR = "artifacts/gradcam"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = load_model(MODEL_PATH, device=device)
    model.eval()

    # Find a good slice
    volumes = os.listdir(DATA_DIR)
    if not volumes:
        print("No data found!")
        return
    
    # Try to find volume 100 or just use the first one
    vol_id = "100"
    if not any(f"volume_{vol_id}" in v for v in volumes):
        vol_id = volumes[0].split("_slice_")[0].replace("volume_", "")
    
    print(f"Using Volume: {vol_id}")
    tumor_slices = find_tumor_slices(DATA_DIR, vol_id, top_n=1)
    if not tumor_slices:
        print("No tumor slices found!")
        return
    
    best_slice_idx, _, slice_path = tumor_slices[0]
    print(f"Using Slice: {best_slice_idx}")

    # Load image and mask
    image, mask_onehot = load_h5_slice(slice_path)
    input_tensor = preprocess_for_inference(image).to(device)

    # Prediction for reference
    pred_mask, probs = predict(model, image, device=device)

    # Class mappings
    classes = {
        1: "WT",
        2: "TC",
        3: "ET"
    }

    # Select target layer
    # For smp.Unet with resnet encoder, it's typically model.encoder.layer4[-1]
    target_layers = [model.encoder.layer4[-1]]
    
    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)

    # Prep visualization image (FLAIR channel normalized)
    mri = image[:, :, 0]
    mri = (mri - mri.min()) / (mri.max() - mri.min() + 1e-8)
    # GradCAM visualization tool expects float32 RGB
    mri_rgb = np.stack([mri, mri, mri], axis=-1).astype(np.float32)

    for class_idx, class_name in classes.items():
        print(f"Generating Grad-CAM for {class_name}...")
        
        # Create target
        # SemanticSegmentationTarget needs the class index and a mask of where that class is in the prediction
        # Actually, if we want to see where the model is looking to predict this class across the whole image
        # we can provide the class index.
        target_mask = (pred_mask == class_idx).astype(np.float32)
        targets = [SemanticSegmentationTarget(class_idx, target_mask)]
        
        # Run CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        
        # Overlay
        visualization = show_cam_on_image(mri_rgb, grayscale_cam, use_rgb=True)
        
        # Save
        save_path = os.path.join(OUTPUT_DIR, f"gradcam_{class_name.lower()}.png")
        plt.figure(figsize=(8, 8))
        plt.imshow(visualization)
        plt.title(f"Grad-CAM: {class_name} (Volume {vol_id}, Slice {best_slice_idx})")
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved: {save_path}")

    # Save a combined image
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (class_idx, class_name) in enumerate(classes.items()):
        img_path = os.path.join(OUTPUT_DIR, f"gradcam_{class_name.lower()}.png")
        img = plt.imread(img_path)
        axes[i].imshow(img)
        axes[i].axis('off')
    
    combined_path = os.path.join(OUTPUT_DIR, "gradcam_combined.png")
    plt.tight_layout()
    plt.savefig(combined_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Combined Grad-CAM saved: {combined_path}")

if __name__ == "__main__":
    generate_gradcams()
