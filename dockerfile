# ============================================================
# Dockerfile for MAST → nnDetection conversion on SOL cluster
# ============================================================

# Base image provided by Radboudumc DIAG
FROM dockerdex.umcn.nl:5005/diag/base-images:base-pt2.7.1

# Working directory inside the container
ARG CODE_DIR="/home/user/source"
WORKDIR ${CODE_DIR}

# Install git (needed to pull your repository)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# ============================================================
# Clone your updated repository
# ============================================================
RUN git clone --depth 1 \
    https://github.com/merlijn-h073/stage_radboudumc_mast_conversion.git \
    ${CODE_DIR}/MAST_conversion

# ============================================================
# Install Python dependencies required by Convert_MAST_to_nnDetection.py
# ============================================================
RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    nibabel \
    tqdm \
    SimpleITK \
    scikit-image

# ============================================================
# Ensure Python can import your repo
# ============================================================
ENV PYTHONPATH="${PYTHONPATH}:${CODE_DIR}:${CODE_DIR}/MAST_conversion"

# Default entrypoint
ENTRYPOINT ["/bin/bash"]
