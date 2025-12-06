# src/preprocess_chaos_dicom.py
import os
import glob
import random
import numpy as np
import cv2
import pydicom
from tqdm import tqdm

CHAOS_MR_ROOT = r"C:\Users\rudre\OneDrive\Desktop\Projects\liver_mri_project\data\raw\chaos-combined-ct-mr-healthy-abdominal-organ\CHAOS_Train_Sets\Train_Sets\MR"
OUT_DIR = "data/processed"
IMG_SIZE = 256
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
LIVER_MIN = 55
LIVER_MAX = 70
RANDOM_SEED = 42
random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def normalize_image(x):
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-6: return np.zeros_like(x, dtype=np.uint8)
    x = (x - mn) / (mx - mn)
    x = (x * 255).astype(np.uint8)
    return x
def extract_liver_mask(gt):
    liver = ((gt >= LIVER_MIN) & (gt <= LIVER_MAX)).astype(np.uint8) * 255
    return liver
def save_pair(image, mask, patient_id, slice_idx, split):
    img_out_dir = os.path.join(OUT_DIR, split, "images"); ensure_dir(img_out_dir)
    mask_out_dir = os.path.join(OUT_DIR, split, "masks"); ensure_dir(mask_out_dir)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    fn = f"{patient_id}_slice_{slice_idx:03d}.png"
    cv2.imwrite(os.path.join(img_out_dir, fn), image)
    cv2.imwrite(os.path.join(mask_out_dir, fn), mask)

def get_patient_ids():
    root = os.path.abspath(CHAOS_MR_ROOT)
    entries = os.listdir(root)
    patients = [d for d in entries if d.isdigit() and os.path.isdir(os.path.join(root, d))]
    return sorted(patients, key=lambda x: int(x))

def split_patients(patient_ids):
    n = len(patient_ids); idx = np.arange(n); np.random.shuffle(idx)
    train_end = int(TRAIN_SPLIT * n); val_end = int((TRAIN_SPLIT + VAL_SPLIT) * n)
    mapping = {}
    for i in idx[:train_end]: mapping[patient_ids[i]] = "train"
    for i in idx[train_end:val_end]: mapping[patient_ids[i]] = "val"
    for i in idx[val_end:]: mapping[patient_ids[i]] = "test"
    return mapping

def process_patient(patient_id, split):
    patient_dir = os.path.join(CHAOS_MR_ROOT, patient_id, "T1DUAL")
    inphase_dir = os.path.join(patient_dir, "DICOM_anon", "InPhase")
    ground_dir = os.path.join(patient_dir, "Ground")
    if not os.path.isdir(inphase_dir) or not os.path.isdir(ground_dir):
        print(f"[WARN] skipping {patient_id}, missing InPhase or Ground")
        return
    dcm_files = sorted(glob.glob(os.path.join(inphase_dir, "*.dcm")))
    gt_files = sorted(glob.glob(os.path.join(ground_dir, "*.png")))
    if len(dcm_files) == 0 or len(gt_files) == 0:
        print(f"[WARN] no files for {patient_id}")
        return
    num_slices = min(len(dcm_files), len(gt_files))
    for i in range(num_slices):
        ds = pydicom.dcmread(dcm_files[i])
        img = ds.pixel_array
        img = normalize_image(img)
        gt = cv2.imread(gt_files[i], cv2.IMREAD_GRAYSCALE)
        if gt is None:
            print("[WARN] failed to read gt", gt_files[i]); continue
        liver_mask = extract_liver_mask(gt)
        save_pair(img, liver_mask, f"patient{patient_id}", i, split)

def main():
    patients = get_patient_ids()
    if not patients: 
        print("No patient folders found under:", CHAOS_MR_ROOT); return
    print("Found patients:", patients)
    splits = split_patients(patients)
    for pid in tqdm(patients, desc="Processing patients"):
        process_patient(pid, splits[pid])
    print("Preprocessing finished. Check data/processed/")

if __name__ == "__main__":
    main()
