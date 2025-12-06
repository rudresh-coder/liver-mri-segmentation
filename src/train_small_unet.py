# src/train_small_unet.py
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LiverMRIDataset
from models.small_unet import SmallUNet
from utils import dice_loss, dice_coef, iou_score, save_checkpoint, set_seed

def train_unet(
    data_root="data/processed",
    img_size=256,
    batch_size=2,
    num_epochs=50,
    lr=1e-3,
    num_workers=0
):
    device = torch.device("cpu")  # CPU only

    print("Using device:", device)
    set_seed(42)

    # 1) Datasets & loaders
    train_dataset = LiverMRIDataset(root=data_root, split="train", img_size=img_size)
    val_dataset   = LiverMRIDataset(root=data_root, split="val",   img_size=img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 2) Model, loss, optimizer
    model = SmallUNet(in_channels=1, out_channels=1).to(device)
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_dice = 0.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": []
    }

    # 3) Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        # show a progress bar over training batches
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss_bce = bce(logits, masks)
            probs = torch.sigmoid(logits)
            loss_d = dice_loss(probs, masks)
            loss = loss_bce + loss_d

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)

        epoch_loss /= len(train_dataset)
        history["train_loss"].append(epoch_loss)

        # 4) Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss_bce = bce(logits, masks)
                probs = torch.sigmoid(logits)
                loss_d = dice_loss(probs, masks)
                loss = loss_bce + loss_d

                val_loss += loss.item() * images.size(0)
                # metrics use thresholded predictions
                preds_bin = (probs > 0.5).float()

                val_dice += dice_coef(preds_bin, masks).item() * images.size(0)
                val_iou  += iou_score(preds_bin, masks).item() * images.size(0)

        val_loss /= len(val_dataset)
        val_dice /= len(val_dataset)
        val_iou  /= len(val_dataset)

        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{num_epochs} "
              f"- train_loss: {epoch_loss:.4f} "
              f"- val_loss: {val_loss:.4f} "
              f"- val_dice: {val_dice:.4f} "
              f"- val_iou: {val_iou:.4f} "
              f"- time: {elapsed:.1f}s")

        # Save best model by Dice
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "checkpoints/small_unet_best.pth")
            print(f"  -> New best model saved (Dice={best_val_dice:.4f})")

    print("Training finished. Best Val Dice:", best_val_dice)

    # Optional: save history to a file for plotting later
    torch.save(history, os.path.join("checkpoints", "small_unet_history.pt"))

if __name__ == "__main__":
    train_unet()
