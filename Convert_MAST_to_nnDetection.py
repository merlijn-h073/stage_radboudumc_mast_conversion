#!/usr/bin/env python3
"""
Convert the MAST dataset to a lung-only nnDetection dataset (voxel-based version, minimal logging).
- Keeps only lesions with lesion_type == 'Lung'.
- Determines lesion presence by checking voxel IDs in masks.
- Processes both baseline (BL) and follow-up (FU) scans independently.
- Renumbers lung lesion voxel values consecutively (1..N) with no gaps.
- No resampling allowed — raises error if shapes differ.
- Creates nnDetection-ready dataset.json and runs validation.
"""

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import nibabel as nib
from tqdm import tqdm

# === PATHS ===
SOURCE_INPUTS = Path(os.environ["SOURCE_INPUTS"])
SOURCE_TARGETS = Path(os.environ["SOURCE_TARGETS"])
TARGET_ROOT = Path(os.environ["TARGET_ROOT"])

IMAGES_DIR = TARGET_ROOT / "imagesTr"
LABELS_DIR = TARGET_ROOT / "labelsTr"
for d in [IMAGES_DIR, LABELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === DATASET METADATA ===
TASK_NAME = "Task_LungLesions"
DESCRIPTION = "Filtered subset of MAST containing only lung lesions (BL + FU, voxel-based presence check)."
REFERENCE = "https://fdat.uni-tuebingen.de/records/75kj1-64747"

# === HELPERS ===
def nn_name(filename: str) -> str:
    """Append _0000.nii.gz if missing for nnDetection."""
    if filename.endswith(".nii.gz"):
        filename = filename[:-7]
    elif filename.endswith(".nii"):
        filename = filename[:-4]
    return filename + "_0000.nii.gz"

def find_file(base: Path, pattern: str):
    matches = list(base.glob(pattern))
    return matches[0] if matches else None

# === MAIN PROCESSING ===
cases = []
kept_log = []
num_patients = 0
num_cases = 0
num_lesions_total = 0

print("\n=== Starting MAST → nnDetection conversion ===")

for csv_path in tqdm(sorted(SOURCE_INPUTS.glob("*.csv")), desc="Processing patients"):
    patient_id = csv_path.stem.split("_")[0]
    num_patients += 1

    df = pd.read_csv(csv_path)
    if "lesion_id" not in df.columns or "lesion_type" not in df.columns:
        continue

    # Select lung lesions
    df["lesion_type"] = df["lesion_type"].astype(str).str.strip().str.lower()
    lung_df = df[df["lesion_type"] == "lung"]
    if lung_df.empty:
        continue

    # Prepare both scan types
    for scan_type in ["BL", "FU"]:
        img_file = find_file(SOURCE_INPUTS, f"{patient_id}_{scan_type}_img_00.nii*")

        # Mask search order depends on scan type
        if scan_type == "BL":
            mask_file = find_file(SOURCE_INPUTS, f"{patient_id}_{scan_type}_mask_00.nii*")
        else:  # FU
            mask_file = find_file(SOURCE_TARGETS, f"{patient_id}_{scan_type}_mask_00.nii*") \
                        or find_file(SOURCE_INPUTS, f"{patient_id}_{scan_type}_mask_00.nii*")

        if not img_file or not mask_file:
            print(f"⚠️ Missing {scan_type} image or mask for {patient_id}.")
            continue

        # Load mask and image
        mask_nii = nib.load(str(mask_file))
        mask_data = mask_nii.get_fdata().astype(np.int32)
        img_nii = nib.load(str(img_file))

        if mask_data.shape != img_nii.shape:
            print(f"⚠️ Shape mismatch for {patient_id}_{scan_type}. Skipping.")
            continue

        # Lesion IDs present in mask
        lesion_ids = lung_df["lesion_id"].astype(int).tolist()
        present_ids = [lid for lid in lesion_ids if (mask_data == lid).any()]
        if not present_ids:
            continue

        new_mask = np.zeros_like(mask_data, dtype=np.int16)
        for new_id, old_id in enumerate(present_ids, start=1):
            new_mask[mask_data == old_id] = new_id

        num_lesions_total += len(present_ids)
        num_cases += 1
        kept_log.append({"patient": patient_id, "scan": scan_type, "count": len(present_ids)})

        # Save filtered mask & image
        mask_out = LABELS_DIR / f"{patient_id}_{scan_type}_mask_00_lung.nii.gz"
        nib.save(nib.Nifti1Image(new_mask, affine=mask_nii.affine, header=mask_nii.header), str(mask_out))

        img_out = IMAGES_DIR / nn_name(f"{patient_id}_{scan_type}_img_00.nii")
        shutil.copyfile(img_file, img_out)

        cases.append({
            "image": str(img_out.relative_to(TARGET_ROOT)),
            "label": str(mask_out.relative_to(TARGET_ROOT))
        })

# === CREATE DATASET.JSON ===
dataset_json = {
    "name": TASK_NAME,
    "description": DESCRIPTION,
    "tensorImageSize": "3D",
    "reference": REFERENCE,
    "licence": "CC BY 4.0",
    "release": "1.0",
    "modality": {"0": "CT"},
    "labels": {"0": "background", "1": "lung_lesion"},
    "numTraining": len(cases),
    "training": cases,
    "test": []
}

with open(TARGET_ROOT / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)

# === FINAL SUMMARY ===
print("\n=== Conversion summary ===")
by_patient = defaultdict(lambda: {"BL": 0, "FU": 0})
for row in kept_log:
    by_patient[row["patient"]][row["scan"]] += row["count"]

for pid, c in by_patient.items():
    print(f" - {pid}: BL={c['BL']}  FU={c['FU']}  (total={c['BL'] + c['FU']})")

print(f"\n✅ Patients processed: {num_patients}")
print(f"✅ Total BL/FU cases exported: {num_cases}")
print(f"✅ Total lung lesions kept: {num_lesions_total}")
print(f"✅ dataset.json created at: {TARGET_ROOT / 'dataset.json'}")

# === VALIDATION BLOCK ===
print("\n=== Validating dataset ===")
valid_count = 0
shape_mismatch = 0
empty_masks = 0
missing_files = 0

for item in cases:
    img_path = TARGET_ROOT / item["image"]
    lbl_path = TARGET_ROOT / item["label"]

    if not img_path.exists() or not lbl_path.exists():
        missing_files += 1
        continue

    img = nib.load(str(img_path))
    lbl = nib.load(str(lbl_path))
    if img.shape != lbl.shape:
        shape_mismatch += 1
        continue

    if not np.any(lbl.get_fdata()):
        empty_masks += 1
        continue

    valid_count += 1

print(f"\nValidation summary:")
print(f"✅ Valid image/mask pairs: {valid_count}")
print(f"⚠️ Shape mismatches: {shape_mismatch}")
print(f"⚠️ Empty masks: {empty_masks}")
print(f"❌ Missing files: {missing_files}")

if shape_mismatch == 0 and empty_masks == 0 and missing_files == 0:
    print("\n✅ All dataset pairs are valid and ready for nnDetection training!")
else:
    print("\n⚠️ Some issues detected — check warnings before training.")
