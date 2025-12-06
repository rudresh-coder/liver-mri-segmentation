import os
import torch
import matplotlib.pyplot as plt

def main():
    hist_path = os.path.join("checkpoints", "att_unet_history.pt")
    if not os.path.exists(hist_path):
        print("History file not found:", hist_path)
        return

    history = torch.load(hist_path)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Attention U-Net: Loss vs Epochs")
    plt.legend()

    # Dice / IoU
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_dice"], label="Val Dice")
    plt.plot(epochs, history["val_iou"], label="Val IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Attention U-Net: Dice / IoU vs Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
