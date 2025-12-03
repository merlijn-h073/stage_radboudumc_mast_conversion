#!/bin/bash
#SBATCH --job-name=mast_convert
#SBATCH --output=logs/mast_convert_%j.out
#SBATCH --error=logs/mast_convert_%j.err
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --gpus-per-task=0

# Use your updated conversion image
#SBATCH --container-image="dockerdex.umcn.nl:5005#merlijn-h073/stage_radboudumc_mast_conversion:2.5"
#SBATCH --container-mounts=/data:/data
#SBATCH --no-container-remap-root

###############################################################################
# CONFIGURATION
###############################################################################
# These must match your conversion script
export SOURCE_INPUTS="/data/bodyct/experiments/merlijnhilberink/inputs"
export SOURCE_TARGETS="/data/bodyct/experiments/merlijnhilberink/targets"
export TARGET_ROOT="/data/bodyct/experiments/merlijnhilberink/Task501_MAST"

echo "=== Starting MAST → nnDetection conversion ==="
echo "SOURCE_INPUTS:   $SOURCE_INPUTS"
echo "SOURCE_TARGETS:  $SOURCE_TARGETS"
echo "TARGET_ROOT:     $TARGET_ROOT"
echo "Docker image:    $SBATCH_CONTAINER_IMAGE"

mkdir -p logs

###############################################################################
# RUN THE PYTHON CONVERSION SCRIPT
###############################################################################
python /workspace/MAST_conversion/Convert_MAST_to_nnDetection.py

echo "=== Conversion complete ==="
