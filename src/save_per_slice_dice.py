import os
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import LiverMRIDataset
from models.unet import UNet
from models.small_unet import SmallUNet
from models.attention_unet import AttentionUNet
from models.unetpp import UNetPP
from matrix import dice_coefficient

# --- Paths (robust no matter where you run from) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
OUT_CSV = CHECKPOINT_DIR / "per_slice_dice.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1  # IMPORTANT: slice-level evaluation

CHECKPOINTS = {
    "UNet": CHECKPOINT_DIR / "unet_best.pth",
    "SmallUNet": CHECKPOINT_DIR / "small_unet_best.pth",
    "AttentionUNet": CHECKPOINT_DIR / "att_unet_best.pth",
    "UNet++": CHECKPOINT_DIR / "unetpp_best.pth",
}


def _build_model(name: str) -> torch.nn.Module:
    if name == "UNet":
        return UNet(n_channels=1, n_classes=1)
    if name == "SmallUNet":
        return SmallUNet(in_channels=1, out_channels=1)
    if name == "AttentionUNet":
        return AttentionUNet(n_channels=1, n_classes=1)
    if name == "UNet++":
        # Keep filters aligned with your training (adjust if your train_unetpp.py used different filters)
        return UNetPP(
            n_channels=1,
            n_classes=1,
            filters=(64, 128, 256, 512, 512),
            deep_supervision=False,
        )
    raise ValueError(f"Unknown model name: {name}")


def _load_weights(model: torch.nn.Module, ckpt_path: Path) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # Handle multiple formats used across your trainers
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            # Sometimes people save the raw state_dict inside a dict with extra keys
            # If it looks like a state_dict already, try it directly.
            state = ckpt
    else:
        state = ckpt

    model.load_state_dict(state)
    return model


def load_model(name: str) -> torch.nn.Module:
    ckpt_path = CHECKPOINTS[name]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = _build_model(name)
    model = _load_weights(model, ckpt_path)
    model.to(DEVICE)
    model.eval()
    return model


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    test_dataset = LiverMRIDataset(root=str(DATA_ROOT), split="test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    rows = []

    for model_name in CHECKPOINTS.keys():
        ckpt_path = CHECKPOINTS[model_name]
        if not ckpt_path.exists():
            print(f"Warning: skipping {model_name} (missing checkpoint: {ckpt_path})")
            continue

        print(f"\nEvaluating {model_name} ({ckpt_path.name}) on DEVICE={DEVICE} ...")
        model = load_model(model_name)

        with torch.no_grad():
            for slice_idx, (images, masks) in enumerate(tqdm(test_loader, desc=f"{model_name}")):
                # images, masks: (1, 1, H, W)
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                logits = model(images)
                # Safety if someone accidentally enabled deep supervision
                if isinstance(logits, (list, tuple)):
                    logits = logits[-1]

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                # dice_coefficient expects 2D tensors (H, W)
                d = dice_coefficient(preds[0, 0].cpu(), masks[0, 0].cpu())

                rows.append(
                    {
                        "model": model_name,
                        "slice_index": slice_idx,
                        "dice": float(d),
                    }
                )

    if not rows:
        raise RuntimeError("No results produced (no checkpoints found or dataset empty).")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved per-slice Dice to: {OUT_CSV}")
    print(df.groupby("model")["dice"].agg(["count", "mean", "std", "min", "max"]).round(4))


if __name__ == "__main__":
    main()
