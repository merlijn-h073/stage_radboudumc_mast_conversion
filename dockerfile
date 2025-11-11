### --------------------------------------------------------------------
### MAST Lung Lesion Conversion – Base Container
### Built on Ducklett, code versioned on GitHub
### --------------------------------------------------------------------

# 1. Base image from the DIAG registry
FROM dockerdex.umcn.nl:5005/diag/base-images:base-pt2.7.1

# 2. Define the working directory
ARG CODE_DIR="/home/user/source"
WORKDIR ${CODE_DIR}

# 3. Install Git (for fetching latest code)
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# 4. Clone the latest version of your GitHub repository
RUN git clone --depth 1 https://github.com/Merlijn-H073/Stage_Radboudumc_MAST_Conversion.git ${CODE_DIR}/MAST_conversion

# 5. Install required Python dependencies
RUN pip3 install --no-cache-dir numpy pandas nibabel tqdm SimpleITK scikit-image

# 6. Set environment variables for Python path
ENV PYTHONPATH="${PYTHONPATH}:${CODE_DIR}:${CODE_DIR}/MAST_conversion:/opt/ASAP/bin"

# 7. Default behavior — start an interactive shell
ENTRYPOINT ["/bin/bash"]

