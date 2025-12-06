# src/plot_history.py
import os
import torch
import matplotlib.pyplot as plt

def main():
    hist_path = os.path.join("checkpoints", "unet_history.pt")
    if not os.path.exists(hist_path):
        print("History file not found:", hist_path)
        return

    history = torch.load(hist_path)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss vs Epochs")

    plt.subplot(1,2,2)
    plt.plot(epochs, history["val_dice"], label="Val Dice")
    plt.plot(epochs, history["val_iou"], label="Val IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Dice / IoU vs Epochs")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
