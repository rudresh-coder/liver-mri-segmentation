# Liver MRI Segmentation with U‑Net Variants

This repository contains code for preprocessing, training, and evaluating deep learning models for **liver segmentation on abdominal MRI**.  
It supports multiple architectures:

- Standard U‑Net
- Small U‑Net (lightweight, CPU‑friendly)
- Attention U‑Net

The code is designed to be **reproducible**, **research‑oriented**, and easy to extend.


## Repository Structure

liver_mri_project/
├── .gitignore
├── README.md
├── checkpoints/               # Saved model weights & training history (not tracked in git)
│   ├── small_unet_best.pth
│   ├── unet_best.pth
│   └── unet_history.pt
├── data/
│   ├── raw/                   # Raw CHAOS data (NOT in repo; download separately)
│   └── processed/             # Preprocessed PNG images & masks (created by preprocessing script)
│       ├── train/
│       │   ├── images/
│       │   └── masks/
│       ├── val/
│       │   ├── images/
│       │   └── masks/
│       └── test/
│           ├── images/
│           └── masks/
├── src/
│   ├── check_chaos_paths.py   # Utility to verify CHAOS folder structure
│   ├── dataloader_test.py     # Quick sanity‑check for dataset & DataLoader
│   ├── dataset.py             # LiverMRIDataset for loading images & masks
│   ├── models/
│   │   ├── unet.py            # Standard U‑Net
│   │   ├── small_unet.py      # Small U‑Net
│   │   └── attention_unet.py  # Attention U‑Net (if present)
│   ├── plot_history.py                # Plot training history (base UNet)
│   ├── plot_history_small_unet.py     # Plot history for Small U‑Net
│   ├── plot_history_attention_unet.py # Plot history for Attention U‑Net
│   ├── preprocess_chaos_dicom.py      # DICOM → PNG preprocessing for CHAOS MRI
│   ├── train_unet.py                  # Train standard U‑Net
│   ├── train_small_unet.py            # Train Small U‑Net
│   ├── train_attention_unet.py        # Train Attention U‑Net
│   └── utils.py               # Metrics (Dice, IoU), losses, checkpoint helpers, seeding
└── venv/                      # Local virtual environment (ignored in git)

## Important

The CHAOS dataset and generated processed data are not included in this repository and are excluded via .gitignore.
You must download the dataset yourself and run the preprocessing script.

## Dataset

This project is built around the CHAOS Challenge MRI dataset:
  > CHAOS: Combined (CT‑MR) Healthy Abdominal Organ Segmentation
  > Official site: https://chaos.grand-challenge.org/

Not included in this repo

Due to licensing and size constraints, raw DICOM data and processed PNG slices are not pushed to GitHub.

## Set up

1. Clone the repository
2. Create and activate a virtual environment : 
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

If not, install the key packages manually:
pip install torch torchvision torchaudio
pip install numpy opencv-python pydicom matplotlib tqdm