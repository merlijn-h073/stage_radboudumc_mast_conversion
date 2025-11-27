#!/usr/bin/env python3
"""
Convert the MAST dataset to a lung-only nnDetection dataset (voxel-based version, minimal logging).
- Keeps only lesions with lesion_type == 'Lung'.
- Determines lesion presence by checking voxel IDs in masks.
- Processes both baseline (BL) and follow-up (FU) scans independently.
- Renumbers lung lesion voxel values consecutively (1..N) with no gaps.
- No resampling allowed — raises error if shapes differ.
- Creates nnDetection-ready dataset.json and runs validation.
Convert the MAST dataset to an nnDetection-compatible lung-lesion dataset
with automatic train/test splitting based on a predefined patient-ID list.
- Automatically routes cases into imagesTr/labelsTr OR imagesTs/labelsTs
- Builds dataset.json with correct "training" and "test" lists
- Writes instance metadata for each mask (nnDetection requirement)
- Validation included
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

# =====================================================================================
# PATHS
# =====================================================================================

SOURCE_INPUTS = Path(os.environ["SOURCE_INPUTS"])
SOURCE_TARGETS = Path(os.environ["SOURCE_TARGETS"])
TARGET_ROOT = Path(os.environ["TARGET_ROOT"])

IMAGES_TR = TARGET_ROOT / "imagesTr"
LABELS_TR = TARGET_ROOT / "labelsTr"
IMAGES_TS = TARGET_ROOT / "imagesTs"
LABELS_TS = TARGET_ROOT / "labelsTs"

for d in [IMAGES_TR, LABELS_TR, IMAGES_TS, LABELS_TS]:
    d.mkdir(parents=True, exist_ok=True)

# =====================================================================================
# TEST SET (Provided)
# =====================================================================================

TEST_IDS = {
    "9872ed9fc2","3b8a614226","6883966fd8","74db120f0a","8f121ce07d",
    "a27ef1c79c","7a614fd06c","5ef059938b","c0e190d826","555d6702c9",
    "2a79ea27c2","96da2f590c","ce08becc73","3644a684f9","e46d0ac1f1",
    "c4de9fe968","5072d348b2","dc5c7986da","fb8490a950","18bbcaa5ef",
    "5737c6ec2e","6c4b761a28","e64770ac6b","441954d29a","defd402043",
    "cfa0860e83","0f49c89d1e","e7db56dbec","19f3cd308f","57aeee35c9",
    "93dd4de5cd","c8ffe9a587","5f93f98352","13fe9d8431","58a2fc6ed3"
}

# =====================================================================================
# HELPERS
# =====================================================================================

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


# =====================================================================================
# MAIN PROCESSING
# =====================================================================================

print("\n=== Starting MAST → nnDetection conversion (with train/test split) ===")

train_cases = []
test_cases = []

kept_log = []
num_patients = 0
num_cases = 0
num_lesions_total = 0

for csv_path in tqdm(sorted(SOURCE_INPUTS.glob("*.csv")), desc="Processing patients"):
    patient_id = csv_path.stem.split("_")[0]
    num_patients += 1

    is_test = patient_id in TEST_IDS

    df = pd.read_csv(csv_path)
    if "lesion_id" not in df.columns or "lesion_type" not in df.columns:
        continue

    df["lesion_type"] = df["lesion_type"].astype(str).str.strip().str.lower()
    lung_df = df[df["lesion_type"] == "lung"]
    if lung_df.empty:
        continue

    # Process both BL and FU scans
    for scan_type in ["BL", "FU"]:
        img_file = find_file(SOURCE_INPUTS, f"{patient_id}_{scan_type}_img_00.nii*")

        # mask lookup
        if scan_type == "BL":
            mask_file = find_file(SOURCE_INPUTS, f"{patient_id}_{scan_type}_mask_00.nii*")
        else:  # FU
            mask_file = find_file(SOURCE_TARGETS, f"{patient_id}_{scan_type}_mask_00.nii*") \
                        or find_file(SOURCE_INPUTS, f"{patient_id}_{scan_type}_mask_00.nii*")

        if not img_file or not mask_file:
            print(f"⚠️ Missing {scan_type} image or mask for {patient_id}.")
            continue

        mask_nii = nib.load(str(mask_file))
        mask_data = mask_nii.get_fdata().astype(np.int32)
        img_nii = nib.load(str(img_file))

        if mask_data.shape != img_nii.shape:
            print(f"⚠️ Shape mismatch for {patient_id}_{scan_type}. Skipping.")
            continue

        lesion_ids = lung_df["lesion_id"].astype(int).tolist()
        present_ids = [lid for lid in lesion_ids if (mask_data == lid).any()]
        if not present_ids:
            continue

        # Renumber masks to 1..N
        new_mask = np.zeros_like(mask_data, dtype=np.int16)
        for new_id, old_id in enumerate(present_ids, start=1):
            new_mask[mask_data == old_id] = new_id

        num_lesions_total += len(present_ids)
        num_cases += 1
        kept_log.append({"patient": patient_id, "scan": scan_type, "count": len(present_ids)})

        # Output paths (train or test)
        if is_test:
            img_out = IMAGES_TS / nn_name(f"{patient_id}_{scan_type}_img_00.nii")
            mask_out = LABELS_TS / f"{patient_id}_{scan_type}_mask_00_lung.nii.gz"
        else:
            img_out = IMAGES_TR / nn_name(f"{patient_id}_{scan_type}_img_00.nii")
            mask_out = LABELS_TR / f"{patient_id}_{scan_type}_mask_00_lung.nii.gz"

        # Save image
        shutil.copyfile(img_file, img_out)

        # Save mask
        nib.save(nib.Nifti1Image(new_mask, affine=mask_nii.affine,
                                 header=mask_nii.header),
                 str(mask_out))

        # Instance metadata JSON
        instances = {str(i): 1 for i in range(1, len(present_ids) + 1)}
        json_out = mask_out.with_suffix(".json")
        with open(json_out, "w") as jf:
            json.dump({"instances": instances}, jf, indent=4)

        # Add to dataset.json lists
        rel_img = str(img_out.relative_to(TARGET_ROOT))
        rel_lbl = str(mask_out.relative_to(TARGET_ROOT))

        if is_test:
            test_cases.append({"image": rel_img})
        else:
            train_cases.append({"image": rel_img, "label": rel_lbl})


# =====================================================================================
# CREATE DATASET.JSON
# =====================================================================================

dataset_json = {
    "name": "Task_LungLesions",
    "description": "Filtered subset of MAST (lung lesions only, BL+FU).",
    "tensorImageSize": "3D",
    "reference": "https://fdat.uni-tuebingen.de/records/75kj1-64747",
    "licence": "CC BY 4.0",
    "release": "1.0",
    "modality": {"0": "CT"},
    "labels": {"0": "background", "1": "lung_lesion"},
    "numTraining": len(train_cases),
    "training": train_cases,
    "test": test_cases
}

with open(TARGET_ROOT / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)

# =====================================================================================
# SUMMARY
# =====================================================================================

print("\n=== Conversion Summary ===")
bp = defaultdict(lambda: {"BL": 0, "FU": 0})
for row in kept_log:
    bp[row["patient"]][row["scan"]] += row["count"]

for pid, c in bp.items():
    print(f" - {pid}: BL={c['BL']}  FU={c['FU']}  (total={c['BL'] + c['FU']})")

print(f"\nTotal patients processed: {num_patients}")
print(f"Total BL/FU cases exported: {num_cases}")
print(f"Total lung lesions kept: {num_lesions_total}")
print(f"Train cases: {len(train_cases)}")
print(f"Test cases:  {len(test_cases)}")
print(f"\nDataset.json written to: {TARGET_ROOT / 'dataset.json'}")

# =====================================================================================
# SIMPLE VALIDATION
# =====================================================================================

print("\n=== Validating dataset ===")
valid = 0
missing = 0
empty = 0
shape_mismatch = 0

all_cases = train_cases + test_cases

for item in all_cases:
    img_path = TARGET_ROOT / item["image"]

    if "label" in item:
        lbl_path = TARGET_ROOT / item["label"]
    else:
        # test set: labelsTs may or may not exist
        lbl_try = (TARGET_ROOT / item["image"]).name.replace("_img_00_0000.nii.gz",
                                                             "_mask_00_lung.nii.gz")
        lbl_path = TARGET_ROOT / ("labelsTs" / lbl_try)

    if not img_path.exists() or not lbl_path.exists():
        missing += 1
        continue

    img = nib.load(str(img_path))
    lbl = nib.load(str(lbl_path))

    if img.shape != lbl.shape:
        shape_mismatch += 1
        continue

    if not np.any(lbl.get_fdata()):
        empty += 1
        continue

    valid += 1

print(f"Valid cases: {valid}")
print(f"Missing files: {missing}")
print(f"Empty masks: {empty}")
print(f"Shape mismatches: {shape_mismatch}")

if missing == 0 and empty == 0 and shape_mismatch == 0:
    print("\n✅ Dataset is ready for nnDetection preprocessing.")
else:
    print("\n⚠️ Some issues detected. Inspect warnings.")
