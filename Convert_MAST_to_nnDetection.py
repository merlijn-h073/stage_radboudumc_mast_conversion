#!/usr/bin/env python3
"""
Convert the MAST dataset to a lung-only nnDetection dataset (voxel-based version).

Main features
-------------
- Keeps only lesions with lesion_type == 'Lung'.
- Determines lesion presence by checking voxel IDs in masks.
- Processes both baseline (BL) and follow-up (FU) scans independently.
- Renumbers lung lesion voxel values consecutively (1..N) with no gaps.
- No resampling allowed — skips cases if image and mask shapes differ.
- Automatically performs a train/test split based on a predefined patient-ID list.
- Writes files in nnDetection-compatible naming/layout:
    imagesTr/caseID_0000.nii.gz
    labelsTr/caseID.nii.gz
    labelsTr/caseID.json
    imagesTs/caseID_0000.nii.gz
    labelsTs/caseID.nii.gz
    labelsTs/caseID.json
- Generates nnDetection-style dataset.json
- Runs simple validation at the end.
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
# PATHS (FROM ENVIRONMENT)
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
    """
    Build nnDetection image filename for a single-modality case.
    case_id -> 'case_id_0000.nii.gz'
    """
    return f"{case_id}_0000.nii.gz"


def find_file(base: Path, pattern: str):
    """
    Find first file that matches a glob pattern in 'base', or None.
    """
    matches = list(base.glob(pattern))
    return matches[0] if matches else None


# =====================================================================================
# MAIN PROCESSING
# =====================================================================================

print("\n=== Starting MAST → nnDetection conversion (with train/test split) ===")
print(f"SOURCE_INPUTS = {SOURCE_INPUTS}")
print(f"SOURCE_TARGETS = {SOURCE_TARGETS}")
print(f"TARGET_ROOT    = {TARGET_ROOT}")

train_cases = []
test_cases = []

kept_log = []
num_patients = 0
num_cases = 0
num_lesions_total = 0

for csv_path in tqdm(sorted(SOURCE_INPUTS.glob("*.csv")), desc="Processing patients"):
    # Expect filenames like '013d407166_BL_inputs.csv' → patient_id = '013d407166'
    patient_id = csv_path.stem.split("_")[0]
    num_patients += 1

    is_test = patient_id in TEST_IDS

    df = pd.read_csv(csv_path)
    if "lesion_id" not in df.columns or "lesion_type" not in df.columns:
        continue

    df["lesion_type"] = df["lesion_type"].astype(str).str.strip().str.lower()
    lung_df = df[df["lesion_type"] == "lung"]
    if lung_df.empty:
        # No lung lesions → skip patient
        continue

    # Process both BL and FU scans independently
    for scan_type in ["BL", "FU"]:
        # Image is always in SOURCE_INPUTS
        img_file = find_file(SOURCE_INPUTS, f"{patient_id}_{scan_type}_img_00.nii*")

        # Mask: BL in SOURCE_INPUTS; FU possibly in SOURCE_TARGETS (post-processed)
        if scan_type == "BL":
            mask_file = find_file(SOURCE_INPUTS, f"{patient_id}_{scan_type}_mask_00.nii*")
        else:
            mask_file = (
                find_file(SOURCE_TARGETS, f"{patient_id}_{scan_type}_mask_00.nii*")
                or find_file(SOURCE_INPUTS, f"{patient_id}_{scan_type}_mask_00.nii*")
            )

        if not img_file or not mask_file:
            print(f"⚠️ Missing {scan_type} image or mask for {patient_id}. Skipping this scan.")
            continue

        # Load and check shapes
        mask_nii = nib.load(str(mask_file))
        mask_data = mask_nii.get_fdata().astype(np.int32)
        img_nii = nib.load(str(img_file))

        if mask_data.shape != img_nii.shape:
            print(f"⚠️ Shape mismatch for {patient_id}_{scan_type}. Skipping this scan.")
            continue

        # Determine which lung lesion IDs are actually present in the mask
        lesion_ids = lung_df["lesion_id"].astype(int).tolist()
        present_ids = [lid for lid in lesion_ids if (mask_data == lid).any()]
        if not present_ids:
            # Patient has lung lesions, but none visible in this scan
            continue

        # Renumber mask to 1..N (instance labels)
        new_mask = np.zeros_like(mask_data, dtype=np.int16)
        for new_id, old_id in enumerate(present_ids, start=1):
            new_mask[mask_data == old_id] = new_id

        num_lesions_total += len(present_ids)
        num_cases += 1
        kept_log.append({"patient": patient_id, "scan": scan_type, "count": len(present_ids)})

        # Case ID used by nnDetection (and for dataset.json)
        case_id = f"{patient_id}_{scan_type}"

        # Select output dirs depending on train/test split
        if is_test:
            img_out = IMAGES_TS / nn_image_name(case_id)
            mask_out = LABELS_TS / f"{case_id}.nii.gz"
            json_out = LABELS_TS / f"{case_id}.json"
        else:
            img_out = IMAGES_TR / nn_image_name(case_id)
            mask_out = LABELS_TR / f"{case_id}.nii.gz"
            json_out = LABELS_TR / f"{case_id}.json"

        # Save image (copy the original NIfTI)
        shutil.copyfile(img_file, img_out)

        # Save instance mask
        nib.save(
            nib.Nifti1Image(new_mask, affine=mask_nii.affine, header=mask_nii.header),
            str(mask_out),
        )

        # Instance metadata JSON: each label 1..N is a lesion instance
        instances = {str(i): 1 for i in range(1, len(present_ids) + 1)}
        with open(json_out, "w") as jf:
            json.dump({"instances": instances}, jf, indent=4)

        # Add to dataset.json lists (relative paths)
        rel_img = str(img_out.relative_to(TARGET_ROOT))
        rel_lbl = str(mask_out.relative_to(TARGET_ROOT))

        if is_test:
            # For nnDetection, test entries usually have only "image";
            # labelsTs still exist, but are not referenced here.
            test_cases.append({"image": rel_img})
        else:
            train_cases.append({"image": rel_img, "label": rel_lbl})

# =====================================================================================
# CREATE DATASET.JSON (nnDetection-STYLE)
# =====================================================================================

dataset_json = {
    # nnDetection-required keys
    "task": "Task501_MAST",        
    "name": "Task_LungLesions",    
    "dim": 3,                      # 3D data
    "target_class": 1,             # lesion class of interest
    "test_labels": True,           # we  have labels for test set (labelsTs)

    # Labels and modalities
    "labels": {
        "0": "background",
        "1": "lung_lesion"
    },
    "modalities": {
        "0": "CT"
    },

    # nnUNet-style metadata
    "tensorImageSize": "3D",
    "reference": "https://fdat.uni-tuebingen.de/records/75kj1-64747",
    "licence": "CC BY 4.0",
    "release": "1.0",

    # Training and test splits
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
bp = defaultdict(lambda: {"BL": 0, "FU": 0})
for row in kept_log:
    bp[row["patient"]][row["scan"]] += row["count"]

for pid, c in bp.items():
    print(f" - {pid}: BL={c['BL']}  FU={c['FU']}  (total={c['BL'] + c['FU']})")

print(f"\nTotal patients processed:         {num_patients}")
print(f"Total BL/FU cases exported:       {num_cases}")
print(f"Total lung lesions kept (voxels): {num_lesions_total}")
print(f"Train cases in dataset.json:      {len(train_cases)}")
print(f"Test cases in dataset.json:       {len(test_cases)}")
print(f"\nDataset.json written to: {TARGET_ROOT / 'dataset.json'}")

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

    if "label" in item:
        # Training case: label path is explicitly in dataset.json
        lbl_path = TARGET_ROOT / item["label"]
    else:
        # Test case: infer label path from image name
        # Example: imagesTs/013d407166_BL_0000.nii.gz → case_id = 013d407166_BL
        image_rel = Path(item["image"])
        case_id = image_rel.stem.replace("_0000", "")
        lbl_path = TARGET_ROOT / "labelsTs" / f"{case_id}.nii.gz"

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
    print("\n⚠️ Some issues detected. Please inspect the warnings above.")
