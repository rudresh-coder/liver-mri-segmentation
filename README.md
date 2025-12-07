# Liver MRI Segmentation with U‑Net Variants

This repository contains code for preprocessing, training, and evaluating deep learning models for **liver segmentation on abdominal MRI**.  
It supports multiple architectures:

- Standard U‑Net
- Small U‑Net (lightweight, CPU‑friendly)
- Attention U‑Net

The code is designed to be **reproducible**, **research‑oriented**, and easy to extend.

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

# 1. Clone the repository
# 2. Create and activate a virtual environment : 
python -m venv venv

Windows (PowerShell)
venv\Scripts\Activate.ps1

macOS / Linux
source venv/bin/activate

# 3. Install dependencies

pip install -r requirements.txt

If not, install the key packages manually:
pip install torch torchvision torchaudio
pip install numpy opencv-python pydicom matplotlib tqdm