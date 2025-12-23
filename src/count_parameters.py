import sys
from pathlib import Path

import torch

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

    total_params = 0
    for name, model in models.items():
        model = model.to(device)
        model.eval()

        params = count_parameters(model)
        total_params += params

        with torch.no_grad():
            y = model(x)
            # UNet++ deep supervision safety (even though deep_supervision=False here)
            if isinstance(y, (list, tuple)):
                y = y[-1]

        print(
            f"{name:15s}: "
            f"{params/1e6:8.3f} M params | "
            f"~{sizeof_mb(model):6.1f} MB params | "
            f"out={tuple(y.shape)}"
        )

    print(f"\nTotal trainable params across listed models: {total_params/1e6:.3f} M\n")


if __name__ == "__main__":
    main()
