# src/plot_history_unetpp.py
import os
import torch
import matplotlib.pyplot as plt

CHECKPOINT_DIR = "checkpoints"
HIST_PATH = os.path.join(CHECKPOINT_DIR, "unetpp_history.pt")
OUT_FIG = os.path.join(CHECKPOINT_DIR, "unetpp_history.png")


def main():
    if not os.path.isfile(HIST_PATH):
        print(f"History file not found: {HIST_PATH}")
        return

    history = torch.load(HIST_PATH, map_location="cpu")

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_dice = history.get("val_dice", [])
    val_iou = history.get("val_iou", [])

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("UNet++: Loss vs Epochs")
    plt.legend()
    plt.grid(True)

    # Dice / IoU curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_dice, label="Val Dice")
    plt.plot(epochs, val_iou, label="Val IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("UNet++: Dice / IoU vs Epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    print(f"Saved plot to {OUT_FIG}")
    plt.show()


if __name__ == "__main__":
    main()
