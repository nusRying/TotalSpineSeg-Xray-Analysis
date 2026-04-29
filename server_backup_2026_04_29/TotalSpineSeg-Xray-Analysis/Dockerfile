# Use official PyTorch image with CUDA 12.4 support
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Set environment variables for nnU-Net
ENV nnUNet_raw="/app/data/nnUNet/raw"
ENV nnUNet_preprocessed="/app/data/nnUNet/preprocessed"
ENV nnUNet_results="/app/data/nnUNet/results"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirement files first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source code
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/data/nnUNet/raw /app/data/nnUNet/preprocessed /app/data/nnUNet/results

# Default command
ENTRYPOINT ["totalspineseg_xray_inference"]
CMD ["--help"]
