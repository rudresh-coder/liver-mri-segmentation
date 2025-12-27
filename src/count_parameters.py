import sys
from pathlib import Path

import torch
import pandas as pd

# Ensure imports work whether you run from project root or from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if not SRC_DIR.exists():
    raise RuntimeError(f"Expected src directory at: {SRC_DIR}")
sys.path.insert(0, str(SRC_DIR))

from models.unet import UNet
from models.small_unet import SmallUNet
from models.attention_unet import AttentionUNet
from models.unetpp import UNetPP


def count_parameters(model: torch.nn.Module) -> int:
    """Trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sizeof_mb(model: torch.nn.Module) -> float:
    """Approx size of parameters in MB (float32 assumption)."""
    num_params = sum(p.numel() for p in model.parameters())
    return (num_params * 4) / (1024**2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build models using the correct constructors from your workspace
    models = {
        "UNet": UNet(n_channels=1, n_classes=1),
        "SmallUNet": SmallUNet(in_channels=1, out_channels=1),
        "AttentionUNet": AttentionUNet(n_channels=1, n_classes=1),
        "UNet++": UNetPP(
            n_channels=1,
            n_classes=1,
            filters=(64, 128, 256, 512, 512),
            deep_supervision=False,
        ),
    }

    # Optional: quick forward-pass sanity check
    x = torch.randn(1, 1, 256, 256, device=device)

    print(f"\nDevice: {device}")
    print("Model Parameter Count (Trainable Only):\n")

    rows = []
    for name, model in models.items():
        model = model.to(device)
        params_m = count_parameters(model) / 1e6
        mb = sizeof_mb(model)

        display_name = {
            "UNet": "U-Net",
            "SmallUNet": "Small U-Net",
            "AttentionUNet": "Attention U-Net",
            "UNet++": "UNet++",
        }.get(name, name)

        rows.append((display_name, params_m, mb))

    print("\n| Model           | Trainable Parameters (M) | Memory Footprint |")
    print("| --------------- | ------------------------ | ---------------- |")
    for model_name, params_m, mb in rows:
        print(f"| {model_name:<15} | {params_m:>6.2f} M                  | ~{mb:>5.1f} MB         |")

    print(f"\nTotal trainable params across listed models: {sum(row[1] for row in rows):.3f} M\n")

    # Save model size table to checkpoints/
    out_csv = PROJECT_ROOT / "checkpoints" / "model_size_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows, columns=["Model", "Trainable Parameters (M)", "Memory Footprint (MB)"])
    df["Trainable Parameters (M)"] = df["Trainable Parameters (M)"].round(2)
    df["Memory Footprint (MB)"] = df["Memory Footprint (MB)"].round(1)
    df.to_csv(out_csv, index=False)

    print(f"Saved model size summary CSV to: {out_csv}")


if __name__ == "__main__":
    main()
