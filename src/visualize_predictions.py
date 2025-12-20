import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pathlib import Path

from models.unet import UNet
from models.small_unet import SmallUNet
from models.attention_unet import AttentionUNet
from models.unetpp import UNetPP

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "src" / "outputs" / "qualitative"

DEVICE = torch.device("cpu")

IMAGE_DIR = DATA_DIR / "test" / "images"
MASK_DIR  = DATA_DIR / "test" / "masks"

NUM_SAMPLES = 12   

os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),  # (1, H, W)
])

def load_model_weights(model, ckpt_path):
    """Load model weights handling different checkpoint formats"""
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    return model

def load_model(model, weight_path):
    if not os.path.exists(weight_path):
        print(f"Warning: Checkpoint not found: {weight_path}")
        return None
    
    try:
        model = load_model_weights(model, weight_path)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading {weight_path}: {e}")
        return None


models = {
    "UNet": load_model(UNet(n_channels=1, n_classes=1), CHECKPOINT_DIR / "unet_best.pth"),
    "SmallUNet": load_model(SmallUNet(in_channels=1, out_channels=1), CHECKPOINT_DIR / "small_unet_best.pth"),
    "AttentionUNet": load_model(AttentionUNet(n_channels=1, n_classes=1), CHECKPOINT_DIR / "att_unet_best.pth"),
    "UNet++": load_model(UNetPP(n_channels=1, n_classes=1, filters=(64, 128, 256, 512, 512), deep_supervision=False), CHECKPOINT_DIR / "unetpp_best.pth"),
}


models = {name: model for name, model in models.items() if model is not None}

if not models:
    print("Error: No models loaded successfully. Check checkpoint paths.")
    exit(1)

print(f"Loaded models: {list(models.keys())}")


def load_image(path):
    img = Image.open(path).convert("L")
    return transform(img).unsqueeze(0)  # (1, 1, H, W)

if not os.path.exists(IMAGE_DIR):
    print(f"Error: Test image directory not found: {IMAGE_DIR}")
    exit(1)

if not os.path.exists(MASK_DIR):
    print(f"Error: Test mask directory not found: {MASK_DIR}")
    exit(1)


image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.png')])

if len(image_files) == 0:
    print(f"No PNG files found in {IMAGE_DIR}")
    exit(1)

image_files = image_files[:NUM_SAMPLES]
num_models = len(models)

for idx, img_name in enumerate(image_files):
    img_path = os.path.join(IMAGE_DIR, img_name)
    mask_path = os.path.join(MASK_DIR, img_name)
    
    if not os.path.exists(mask_path):
        print(f"Warning: Ground truth mask not found for {img_name}, skipping...")
        continue

    image = load_image(img_path).to(DEVICE)
    gt_mask = np.array(Image.open(mask_path))

    predictions = {}

    with torch.no_grad():
        for name, model in models.items():
            output = model(image)
            if isinstance(output, list):  
                output = output[-1]
            pred = torch.sigmoid(output)
            pred = (pred > 0.5).float()
            predictions[name] = pred.squeeze().cpu().numpy()

 
    total_plots = 2 + num_models 
    fig, axes = plt.subplots(1, total_plots, figsize=(3 * total_plots, 4))
    
    if total_plots == 1:
        axes = [axes]  
    
    fig.suptitle(f"Test Sample {idx + 1}: {img_name}", fontsize=14)

    axes[0].imshow(image.squeeze().cpu(), cmap="gray")
    axes[0].set_title("MRI")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    for i, (name, pred) in enumerate(predictions.items()):
        axes[i + 2].imshow(pred, cmap="gray")
        axes[i + 2].set_title(name)
        axes[i + 2].axis("off")

    save_path = os.path.join(OUTPUT_DIR, f"sample_{idx + 1:02d}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")

print(f"\nQualitative visualization completed. Generated {len(image_files)} samples.")
