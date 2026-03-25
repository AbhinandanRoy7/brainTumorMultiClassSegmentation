"""
Utility functions for visualization, metrics, and analysis.
Color mapping, overlays, Dice scores, tumor location detection.
"""

import numpy as np
from PIL import Image
import io
import base64


# ─── COLOR MAPPING ────────────────────────────────────────────────
# Class 0: Background → Black (transparent)
# Class 1: Edema / Whole Tumor → Gold (#FFD700)
# Class 2: Tumor Core → Orange (#FF8C00)
# Class 3: Enhancing Tumor → Red (#FF0000)

COLOR_MAP = {
    0: [0, 0, 0],         # Background
    1: [255, 215, 0],     # Edema (WT) - Gold
    2: [255, 140, 0],     # Tumor Core - Orange
    3: [255, 0, 0],       # Enhancing Tumor - Red
}

CLASS_NAMES = {
    0: "Background",
    1: "Whole Tumor (Edema)",
    2: "Tumor Core",
    3: "Enhancing Tumor",
}


def color_map_mask(mask):
    """
    Convert class index mask (H, W) to RGB color image (H, W, 3).
    """
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cls_id, color in COLOR_MAP.items():
        colored[mask == cls_id] = color
    
    return colored


def create_overlay(image_slice, colored_mask, alpha=0.4):
    """
    Overlay colored segmentation mask on MRI slice.
    image_slice: (H, W) single channel MRI
    colored_mask: (H, W, 3) RGB mask
    alpha: mask opacity (0.4 = 40%)
    """
    # Normalize MRI to 0-255
    img = image_slice.copy()
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    img = (img * 255).astype(np.uint8)
    
    # Convert to RGB
    img_rgb = np.stack([img, img, img], axis=-1)
    
    # Overlay only where mask is non-zero
    mask_present = colored_mask.sum(axis=-1) > 0
    overlay = img_rgb.copy()
    overlay[mask_present] = (
        (1 - alpha) * img_rgb[mask_present] + 
        alpha * colored_mask[mask_present]
    ).astype(np.uint8)
    
    return overlay


# ─── METRICS ──────────────────────────────────────────────────────
def compute_dice(pred, target, num_classes=4, smooth=1e-6):
    """Compute per-class Dice scores."""
    dices = {}
    class_labels = {1: "wt", 2: "tc", 3: "et"}
    
    for c in range(1, num_classes):  # Skip background
        pred_c = (pred == c).astype(np.float32)
        target_c = (target == c).astype(np.float32)
        intersection = (pred_c * target_c).sum()
        dice = (2.0 * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
        dices[class_labels[c]] = round(float(dice), 4)
    
    dices["overall"] = round(np.mean(list(dices.values())), 4)
    return dices


# ─── TUMOR LOCATION ──────────────────────────────────────────────
def detect_tumor_location(mask):
    """
    Detect tumor location: Left / Right / Center hemisphere.
    Based on centroid of tumor pixels relative to image midpoint.
    """
    tumor_pixels = np.where(mask > 0)
    
    if len(tumor_pixels[0]) == 0:
        return "No tumor detected", None
    
    # Centroid
    centroid_y = float(np.mean(tumor_pixels[0]))
    centroid_x = float(np.mean(tumor_pixels[1]))
    
    h, w = mask.shape
    mid_x = w / 2
    
    # Determine side
    margin = w * 0.15  # 15% margin for "center"
    if centroid_x < mid_x - margin:
        location = "Right Hemisphere"
    elif centroid_x > mid_x + margin:
        location = "Left Hemisphere"
    else:
        location = "Central Region"
    
    return location, {"x": centroid_x, "y": centroid_y}


# ─── TUMOR STATS ─────────────────────────────────────────────────
def compute_tumor_stats(pred_mask):
    """Compute tumor region statistics."""
    total_pixels = pred_mask.size
    
    stats = {
        "total_pixels": int(total_pixels),
        "tumor_pixels": int((pred_mask > 0).sum()),
        "tumor_percentage": round(float((pred_mask > 0).sum() / total_pixels * 100), 2),
        "wt_pixels": int((pred_mask == 1).sum()),
        "tc_pixels": int((pred_mask == 2).sum()),
        "et_pixels": int((pred_mask == 3).sum()),
        "has_tumor": bool((pred_mask > 0).any()),
    }
    
    # Percentages of tumor regions
    if stats["tumor_pixels"] > 0:
        stats["wt_percent"] = round(stats["wt_pixels"] / stats["tumor_pixels"] * 100, 1)
        stats["tc_percent"] = round(stats["tc_pixels"] / stats["tumor_pixels"] * 100, 1)
        stats["et_percent"] = round(stats["et_pixels"] / stats["tumor_pixels"] * 100, 1)
    else:
        stats["wt_percent"] = 0
        stats["tc_percent"] = 0
        stats["et_percent"] = 0
    
    return stats


# ─── MODEL INSIGHTS ──────────────────────────────────────────────
def generate_insights(dice_scores, tumor_stats, confidence, training_history=None):
    """Generate human-readable model insights."""
    insights = []
    
    # Confidence insight
    if confidence > 0.85:
        insights.append({
            "type": "success",
            "title": "High Confidence",
            "text": f"Model confidence: {confidence:.1%}. Predictions are reliable."
        })
    elif confidence > 0.65:
        insights.append({
            "type": "warning", 
            "title": "Moderate Confidence",
            "text": f"Model confidence: {confidence:.1%}. Some regions may be uncertain."
        })
    else:
        insights.append({
            "type": "error",
            "title": "Low Confidence",
            "text": f"Model confidence: {confidence:.1%}. Predictions should be reviewed carefully."
        })
    
    # Dice score insights
    if dice_scores["overall"] > 0.80:
        insights.append({
            "type": "success",
            "title": "Excellent Segmentation",
            "text": f"Overall Dice: {dice_scores['overall']:.4f}. Model is performing very well."
        })
    elif dice_scores["overall"] > 0.60:
        insights.append({
            "type": "warning",
            "title": "Good Segmentation",
            "text": f"Overall Dice: {dice_scores['overall']:.4f}. Results are acceptable."
        })
    
    # Tumor composition
    if tumor_stats["has_tumor"]:
        dominant = max(
            [("Edema", tumor_stats["wt_percent"]),
             ("Tumor Core", tumor_stats["tc_percent"]),
             ("Enhancing", tumor_stats["et_percent"])],
            key=lambda x: x[1]
        )
        insights.append({
            "type": "info",
            "title": "Tumor Composition",
            "text": f"Dominant region: {dominant[0]} ({dominant[1]:.1f}% of tumor area)."
        })
        
        # ET presence
        if tumor_stats["et_percent"] > 20:
            insights.append({
                "type": "error",
                "title": "Significant Enhancing Region",
                "text": f"Enhancing tumor is {tumor_stats['et_percent']:.1f}% of tumor. This typically indicates high-grade glioma."
            })
    
    # Training history insights
    if training_history:
        train_loss = training_history.get('train_loss', [])
        val_loss = training_history.get('val_loss', [])
        if len(train_loss) > 3 and len(val_loss) > 3:
            if val_loss[-1] < train_loss[-1] * 1.1:
                insights.append({
                    "type": "success",
                    "title": "Good Generalization",
                    "text": "Model is generalizing well (train ≈ validation loss)."
                })
            elif val_loss[-1] > train_loss[-1] * 1.5:
                insights.append({
                    "type": "warning",
                    "title": "Overfitting Detected",
                    "text": "Validation loss is significantly higher than training. Consider regularization."
                })
            
            # Check plateau
            if len(val_loss) >= 5:
                recent = val_loss[-5:]
                if max(recent) - min(recent) < 0.01:
                    insights.append({
                        "type": "info",
                        "title": "Performance Plateau",
                        "text": "Loss has plateaued in recent epochs. Training may be converged."
                    })
    
    return insights


# ─── IMAGE ENCODING ──────────────────────────────────────────────
def numpy_to_base64(img_array, format="PNG"):
    """Convert numpy array to base64 encoded string."""
    if img_array.dtype != np.uint8:
        # Normalize to 0-255
        img_min, img_max = img_array.min(), img_array.max()
        if img_max > img_min:
            img_array = (img_array - img_min) / (img_max - img_min)
        img_array = (img_array * 255).astype(np.uint8)
    
    if len(img_array.shape) == 2:
        img = Image.fromarray(img_array, mode='L')
    else:
        img = Image.fromarray(img_array, mode='RGB')
    
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
