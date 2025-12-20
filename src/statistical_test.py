import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from dataset import LiverMRIDataset
from models.unet import UNet
from models.small_unet import SmallUNet
from models.attention_unet import AttentionUNet
from models.unetpp import UNetPP
from matrix import dice_coefficient

DEVICE = torch.device("cpu")

CHECKPOINTS = {
    "UNet": "checkpoints/unet_best.pth",
    "SmallUNet": "checkpoints/small_unet_best.pth",
    "AttentionUNet": "checkpoints/att_unet_best.pth",
    "UNet++": "checkpoints/unetpp_best.pth",
}

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

def load_model(name):
    if name == "UNet":
        model = UNet(n_channels=1, n_classes=1)
    elif name == "SmallUNet":
        model = SmallUNet(in_channels=1, out_channels=1)
    elif name == "AttentionUNet":
        model = AttentionUNet(n_channels=1, n_classes=1)
    elif name == "UNet++":
        model = UNetPP(n_channels=1, n_classes=1, filters=(64, 128, 256, 512, 512), deep_supervision=False)
    else:
        raise ValueError("Unknown model")

    model = load_model_weights(model, CHECKPOINTS[name])
    model.to(DEVICE)
    model.eval()
    return model

def compute_per_image_dice(model, dataloader):
    scores = []
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            for i in range(images.size(0)):
                # Note: dice_coefficient returns a scalar, no need for .item()
                d = dice_coefficient(preds[i, 0], masks[i, 0])
                scores.append(d)

    return np.array(scores)

def main():
    test_dataset = LiverMRIDataset(root="data/processed", split="test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=2, shuffle=False
    )

    dice_scores = {}

    print("Computing per-image Dice scores...")
    for name in CHECKPOINTS:
        if not os.path.exists(CHECKPOINTS[name]):
            print(f"Checkpoint not found: {CHECKPOINTS[name]}")
            continue
            
        model = load_model(name)
        dice_scores[name] = compute_per_image_dice(model, test_loader)
        print(f"{name}: mean Dice = {dice_scores[name].mean():.4f}")

    # Pairwise Wilcoxon tests vs UNet++
    if "UNet++" not in dice_scores:
        print("UNet++ scores not available for comparison")
        return
        
    results = []
    reference = "UNet++"
    
    for model_name in ["UNet", "SmallUNet", "AttentionUNet"]:
        if model_name not in dice_scores:
            continue
            
        stat, p_value = wilcoxon(
            dice_scores[reference],
            dice_scores[model_name]
        )
        results.append({
            "Comparison": f"{reference} vs {model_name}",
            "p-value": p_value,
            "Significant (p<0.05)": p_value < 0.05
        })

    if results:
        df = pd.DataFrame(results)
        os.makedirs("checkpoints", exist_ok=True)
        df.to_csv("checkpoints/statistical_significance.csv", index=False)

        print("\nStatistical Test Results:")
        print(df)

if __name__ == "__main__":
    main()
