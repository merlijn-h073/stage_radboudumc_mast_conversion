#!/usr/bin/env python3
"""
Convert the MAST dataset to a lung-only nnDetection dataset (voxel-based, single-class).

- Keeps only lung lesions
- Instance masks: 0 = background, 1..N = instances
- Instance JSONs: ALL instances mapped to class 0
- Single-class nnDetection setup
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
# FIXED TEST SET
# =====================================================================================

TEST_IDS = {
    "9872ed9fc2", "3b8a614226", "6883966fd8", "74db120f0a", "8f121ce07d",
    "a27ef1c79c", "7a614fd06c", "5ef059938b", "c0e190d826", "555d6702c9",
    "2a79ea27c2", "96da2f590c", "ce08becc73", "3644a684f9", "e46d0ac1f1",
    "c4de9fe968", "5072d348b2", "dc5c7986da", "fb8490a950", "18bbcaa5ef",
    "5737c6ec2e", "6c4b761a28", "e64770ac6b", "441954d29a", "defd402043",
    "cfa0860e83", "0f49c89d1e", "e7db56dbec", "19f3cd308f", "57aeee35c9",
    "93dd4de5cd", "c8ffe9a587", "5f93f98352", "13fe9d8431", "58a2fc6ed3",
}

# =====================================================================================
# HELPERS
# =====================================================================================

def nn_image_name(case_id: str) -> str:
    return f"{case_id}_0000.nii.gz"

def find_file(base: Path, pattern: str):
    matches = list(base.glob(pattern))
    return matches[0] if matches else None

# =====================================================================================
# MAIN PROCESSING
# =====================================================================================

train_cases = []
test_cases = []

kept_log = []
num_patients = 0
num_cases = 0
num_lesions_total = 0

print("\n=== Starting MAST → nnDetection conversion (single-class) ===")

for csv_path in tqdm(sorted(SOURCE_INPUTS.glob("*.csv")), desc="Processing patients"):
    patient_id = csv_path.stem.split("_")[0]
    num_patients += 1
    is_test = patient_id in TEST_IDS

    df = pd.read_csv(csv_path)
    if {"lesion_id", "lesion_type"} - set(df.columns):
        continue

    df["lesion_type"] = df["lesion_type"].astype(str).str.strip().str.lower()
    lung_df = df[df["lesion_type"] == "lung"]
    if lung_df.empty:
        continue

    for scan_type in ["BL", "FU"]:
        img_file = find_file(SOURCE_INPUTS, f"{patient_id}_{scan_type}_img_00.nii*")

        if scan_type == "BL":
            mask_file = find_file(SOURCE_INPUTS, f"{patient_id}_{scan_type}_mask_00.nii*")
        else:
            mask_file = (
                find_file(SOURCE_TARGETS, f"{patient_id}_{scan_type}_mask_00.nii*")
                or find_file(SOURCE_INPUTS, f"{patient_id}_{scan_type}_mask_00.nii*")
            )

        if not img_file or not mask_file:
            continue

        img_nii = nib.load(str(img_file))
        mask_nii = nib.load(str(mask_file))
        mask_data = mask_nii.get_fdata().astype(np.int32)

        if img_nii.shape != mask_data.shape:
            continue

        lesion_ids = lung_df["lesion_id"].astype(int).tolist()
        present_ids = [lid for lid in lesion_ids if (mask_data == lid).any()]
        if not present_ids:
            continue

        new_mask = np.zeros_like(mask_data, dtype=np.int16)
        for new_id, old_id in enumerate(present_ids, start=1):
            new_mask[mask_data == old_id] = new_id

        num_cases += 1
        num_lesions_total += len(present_ids)
        kept_log.append({"patient": patient_id, "scan": scan_type, "count": len(present_ids)})

        case_id = f"{patient_id}_{scan_type}"

        if is_test:
            img_out = IMAGES_TS / nn_image_name(case_id)
            mask_out = LABELS_TS / f"{case_id}.nii.gz"
            json_out = LABELS_TS / f"{case_id}.json"
        else:
            img_out = IMAGES_TR / nn_image_name(case_id)
            mask_out = LABELS_TR / f"{case_id}.nii.gz"
            json_out = LABELS_TR / f"{case_id}.json"

        shutil.copyfile(img_file, img_out)

        nib.save(
            nib.Nifti1Image(new_mask, affine=mask_nii.affine, header=mask_nii.header),
            str(mask_out),
        )

        # SINGLE-CLASS instance mapping
        instances = {str(i): 0 for i in range(1, len(present_ids) + 1)}
        with open(json_out, "w") as f:
            json.dump({"instances": instances}, f, indent=4)

        rel_img = str(img_out.relative_to(TARGET_ROOT))
        rel_lbl = str(mask_out.relative_to(TARGET_ROOT))

        if is_test:
            test_cases.append({"image": rel_img, "label": rel_lbl})
        else:
            train_cases.append({"image": rel_img, "label": rel_lbl})

# =====================================================================================
# DATASET.JSON
# =====================================================================================

dataset_json = {
    "task": "Task502_MAST",
    "name": "Task_LungLesions",
    "dim": 3,
    "target_class": None,
    "test_labels": True,

    "labels": {
        "0": "lung_lesion"
    },
    "modalities": {
        "0": "CT"
    },

    "tensorImageSize": "3D",
    "reference": "https://fdat.uni-tuebingen.de/records/75kj1-64747",
    "licence": "CC BY 4.0",
    "release": "1.0",

    "numTraining": len(train_cases),
    "training": train_cases,
    "test": test_cases,
}

with open(TARGET_ROOT / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)

# =====================================================================================
# SUMMARY
# =====================================================================================

print("\n=== Conversion Summary ===")
print(f"Patients processed:        {num_patients}")
print(f"Cases exported (BL/FU):    {num_cases}")
print(f"Lung lesions kept:         {num_lesions_total}")
print(f"Training cases:            {len(train_cases)}")
print(f"Test cases:                {len(test_cases)}")

# =====================================================================================
# VALIDATION
# =====================================================================================

print("\n=== Validating dataset ===")

valid = 0
missing = 0
empty = 0
shape_mismatch = 0

all_cases = train_cases + test_cases

for item in all_cases:
    img_path = TARGET_ROOT / item["image"]
    lbl_path = TARGET_ROOT / item["label"]

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

print(f"Valid cases:      {valid}")
print(f"Missing files:    {missing}")
print(f"Empty masks:      {empty}")
print(f"Shape mismatches: {shape_mismatch}")

if missing == 0 and empty == 0 and shape_mismatch == 0:
    print("\n✅ Dataset is ready for nnDetection preprocessing.")
else:
    print("\n⚠️ Dataset has issues — please inspect the counts above.")
