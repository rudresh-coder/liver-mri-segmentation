import os

CHAOS_MR_ROOT = "data/raw/chaos-combined-ct-mr-healthy-abdominal-organ/CHAOS_Train_Sets/Train_Sets/MR"

def main():
    root_abs = os.path.abspath(CHAOS_MR_ROOT)
    print("CHAOS_MR_ROOT (absolute):", root_abs)
    print("Exists?             :", os.path.isdir(root_abs))

    if not os.path.isdir(root_abs):
        return

    print("\nEntries inside MR folder:")
    for name in os.listdir(root_abs):
        full = os.path.join(root_abs, name)
        print("  ", name, "->", "DIR" if os.path.isdir(full) else "FILE")

    # Show one patient example
    example_patient = "10"  # change if you know another ID exists
    patient_dir = os.path.join(root_abs, example_patient, "T1DUAL")
    inphase_dir = os.path.join(patient_dir, "DICOM_anon", "InPhase")
    ground_dir = os.path.join(patient_dir, "Ground")

    print("\nExample patient:", example_patient)
    print("  patient_dir  :", patient_dir, "  exists?", os.path.isdir(patient_dir))
    print("  inphase_dir  :", inphase_dir, "  exists?", os.path.isdir(inphase_dir))
    print("  ground_dir   :", ground_dir, "  exists?", os.path.isdir(ground_dir))

if __name__ == "__main__":
    main()
