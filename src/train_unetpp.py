import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.unetpp import UNetPP

from dataset import LiverMRIDataset  


# ---------- Config ----------
DATA_ROOT = "data/processed"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 2
NUM_EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Metrics ----------
def dice_coeff(pred, target, eps=1e-7):
    """
    pred, target: shape (N, 1, H, W), values in {0,1} for target; pred is sigmoid thresholded.
    """
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()


def iou_coeff(pred, target, eps=1e-7):
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


# ---------- Loss (BCE + soft Dice) ----------
class SoftDiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


# ---------- Train / Eval loops ----------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)  # (B, 1, H, W)
        masks = masks.to(device)    # (B, 1, H, W)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    dices = []
    ious = []

    for images, masks in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        val_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        dice = dice_coeff(preds, masks)
        iou = iou_coeff(preds, masks)
        dices.append(dice)
        ious.append(iou)

    val_loss /= len(loader.dataset)
    mean_dice = sum(dices) / len(dices) if dices else 0.0
    mean_iou = sum(ious) / len(ious) if ious else 0.0

    return val_loss, mean_dice, mean_iou


def main():
    print(f"Using device: {DEVICE}")

    # Datasets and loaders
    train_dataset = LiverMRIDataset(root=DATA_ROOT, split="train")
    val_dataset   = LiverMRIDataset(root=DATA_ROOT, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    model = UNetPP(n_channels=1, n_classes=1, filters=(64, 128, 256, 512, 512), deep_supervision=False)
    model = model.to(DEVICE)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    # History + checkpoint
    best_val_dice = 0.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
    }

    ckpt_path = os.path.join(CHECKPOINT_DIR, "unetpp_best.pth")
    hist_path = os.path.join(CHECKPOINT_DIR, "unetpp_history.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, DEVICE)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch}/{NUM_EPOCHS} - "
            f"train_loss: {train_loss:.4f} - "
            f"val_loss: {val_loss:.4f} - "
            f"val_dice: {val_dice:.4f} - "
            f"val_iou: {val_iou:.4f} - "
            f"time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_dice": best_val_dice,
                    "history": history,
                },
                ckpt_path,
            )
            print(f"  -> New best model saved (Dice={best_val_dice:.4f})")

        # Save history after each epoch (for safety)
        torch.save(history, hist_path)

    print(f"Training finished. Best Val Dice: {best_val_dice}")


if __name__ == "__main__":
    main()
