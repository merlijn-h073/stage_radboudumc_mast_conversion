#!/bin/bash
#SBATCH --job-name=MAST_convert
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/data/bodyct/experiments/merlijnhilberink/logs/mast_convert_%j.out
#SBATCH --error=/data/bodyct/experiments/merlijnhilberink/logs/mast_convert_%j.err

# Create log directory if not exists
mkdir -p /data/bodyct/experiments/merlijnhilberink/logs

# Environment variables for script
export SOURCE_INPUTS=/data/bodyct/experiments/nielsrocholl/MAST/v2/inputsTr
export SOURCE_TARGETS=/data/bodyct/experiments/nielsrocholl/MAST/v2/targetsTr
export TARGET_ROOT=/data/bodyct/experiments/merlijnhilberink/MAST_nnDet

mkdir -p $TARGET_ROOT

# Run the conversion inside your Docker container
srun \
  --container-image="dockerdex.umcn.nl:5005#merlijn-h073/stage_radboudumc_mast_conversion:2.2" \
  --container-mounts=/data:/data \
  --no-container-remap-root \
  python3 /home/user/source/Stage_Radboudumc_MAST_Conversion/Convert_MAST_to_nnDetection.py
