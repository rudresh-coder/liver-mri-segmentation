import torch
import csv
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from torch.utils.data import DataLoader

from dataset import LiverMRIDataset
from matrix import (
    dice_coefficient,
    iou_score,
    precision_score,
    recall_score,
    f1_score,
)

from models.unet import UNet
from models.small_unet import SmallUNet
from models.attention_unet import AttentionUNet
from models.unetpp import UNetPP


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_IMG_DIR = Path("data/processed/test/images")
TEST_MASK_DIR = Path("data/processed/test/masks")
CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_CSV = CHECKPOINT_DIR / "test_metrics.csv"


MODELS = {
    "UNet": {
        "model": UNet(n_channels=1, n_classes=1),  # Correct
        "ckpt": "unet_best.pth",
    },
    "SmallUNet": {
        "model": SmallUNet(in_channels=1, out_channels=1),  # Fixed parameter names
        "ckpt": "small_unet_best.pth",
    },
    "AttentionUNet": {
        "model": AttentionUNet(n_channels=1, n_classes=1),  # Correct
        "ckpt": "att_unet_best.pth",
    },
    "UNet++": {
        "model": UNetPP(n_channels=1, n_classes=1, filters=(64, 128, 256, 512, 512), deep_supervision=False),  # Added required filters
        "ckpt": "unetpp_best.pth",
    },
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


def evaluate_model(model, loader):
    model.eval()

    dice_list, iou_list = [], []
    prec_list, recall_list, f1_list = [], [], []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            for i in range(preds.size(0)):
                p = preds[i, 0]
                t = masks[i, 0]

                dice_list.append(dice_coefficient(p, t))
                iou_list.append(iou_score(p, t))
                prec_list.append(precision_score(p, t))
                recall_list.append(recall_score(p, t))
                f1_list.append(f1_score(p, t))

    return {
        "Dice": sum(dice_list) / len(dice_list),
        "IoU": sum(iou_list) / len(iou_list),
        "Precision": sum(prec_list) / len(prec_list),
        "Recall": sum(recall_list) / len(recall_list),
        "F1": sum(f1_list) / len(f1_list),
    }


def main():
    print(f"Using device: {DEVICE}")

    test_dataset = LiverMRIDataset(root="data/processed", split="test")
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    results = []

    for name, cfg in MODELS.items():
        print(f"\nEvaluating {name}...")

        model = cfg["model"].to(DEVICE)
        ckpt_path = CHECKPOINT_DIR / cfg["ckpt"]

        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}")
            continue

        model = load_model_weights(model, ckpt_path)
        metrics = evaluate_model(model, test_loader)

        row = {"Model": name, **metrics}
        results.append(row)

        print(row)

    if results:
        OUTPUT_CSV.parent.mkdir(exist_ok=True)

        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print(f"\nSaved test metrics to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
