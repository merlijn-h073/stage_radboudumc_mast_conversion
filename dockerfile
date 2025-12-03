# ============================================================
# Local-build compatible Dockerfile for MAST → nnDetection conversion
# ============================================================

# Use a PUBLIC CUDA + PyTorch base image (works on Windows)
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Make sure everything is up to date
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Working directory inside container
ARG CODE_DIR="/workspace"
WORKDIR ${CODE_DIR}

# ============================================================
# Copy your local repository into the container
# (Much faster and avoids GitHub rate limits)
# ============================================================
COPY . ${CODE_DIR}/MAST_conversion/

# ============================================================
# Install Python dependencies required by Convert_MAST_to_nnDetection.py
# ============================================================
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    nibabel \
    tqdm \
    SimpleITK \
    scikit-image

# ============================================================
# Add your conversion package to PYTHONPATH
# ============================================================
ENV PYTHONPATH="${PYTHONPATH}:${CODE_DIR}:${CODE_DIR}/MAST_conversion"

# Default entrypoint
ENTRYPOINT ["/bin/bash"]
