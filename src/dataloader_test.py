import torch
from torch.utils.data import DataLoader
from dataset import LiverMRIDataset
import matplotlib.pyplot as plt
import torchvision

def show_batch(images, masks):
    grid_imgs = torchvision.utils.make_grid(images, nrow=4, normalize=True)
    grid_masks = torchvision.utils.make_grid(masks, nrow=4, normalize=False)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("MRI Images")
    plt.imshow(grid_imgs.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Liver Masks")
    plt.imshow(grid_masks.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis('off')

    plt.show()

def main():
    dataset = LiverMRIDataset(root="data/processed", split="train")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    images, masks = next(iter(loader))
    print("Batch image shape:", images.shape)
    print("Batch mask shape :", masks.shape)

    show_batch(images, masks)

if __name__ == "__main__":
    main()
