# ============================================================
# Dockerfile for MAST → nnDetection conversion (SOL cluster)
# ============================================================

# Base image provided by Radboudumc DIAG
FROM dockerdex.umcn.nl:5005/diag/base-images:base-pt2.7.1

# Working directory inside the container
ARG CODE_DIR="/home/user/source"
WORKDIR ${CODE_DIR}

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# ============================================================
# Copy your updated repository into the image
# ============================================================
COPY . ${CODE_DIR}/MAST_conversion

# ============================================================
# Install Python dependencies
# ============================================================
RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    nibabel \
    tqdm \
    SimpleITK \
    scikit-image

# ============================================================
# Make Python see your repo
# ============================================================
ENV PYTHONPATH="${PYTHONPATH}:${CODE_DIR}:${CODE_DIR}/MAST_conversion"

# Default entrypoint
ENTRYPOINT ["/bin/bash"]
