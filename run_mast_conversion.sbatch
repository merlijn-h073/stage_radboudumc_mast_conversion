#!/bin/bash
#SBATCH --job-name=mast_convert
#SBATCH --output=logs/mast_convert_%j.out
#SBATCH --error=logs/mast_convert_%j.err
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# === Pyxis / Enroot container settings ===
#SBATCH --gpus-per-task=0
#SBATCH --container-image=dockerdex.umcn.nl:5005#merlijn-h073/stage_radboudumc_mast_conversion:2.4
#SBATCH --container-mounts=/data:/data
#SBATCH --no-container-remap-root

# === Environment variables passed into container ===
export SOURCE_INPUTS="/data/bodyct/experiments/nielsrocholl/MAST/v2/inputsTr"
export SOURCE_TARGETS="/data/bodyct/experiments/nielsrocholl/MAST/v2/targetsTr"
export TARGET_ROOT="/data/bodyct/experiments/merlijnhilberink/MAST_nnDet"

# Create logs dir if needed
mkdir -p logs

echo "=== Starting MAST → nnDetection conversion ==="
echo "Using inputs:       $SOURCE_INPUTS"
echo "Using FU targets:   $SOURCE_TARGETS"
echo "Writing to:         $TARGET_ROOT"
echo "Running inside container: $SBATCH_CONTAINER_IMAGE"

# Run conversion inside the container
python /home/user/source/Convert_MAST_to_nnDetection.py

echo "=== Conversion finished ==="
