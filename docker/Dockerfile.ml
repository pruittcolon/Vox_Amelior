# ML Service - Universal ML Agent with CUDA Support
# Using NVIDIA CUDA base image for GPU acceleration
FROM nvidia/cuda:13.1.0-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Create user and directories
RUN useradd -m -u 1000 nemo && \
    mkdir -p /app/data/uploads /app/archive && \
    chown -R nemo:nemo /app

WORKDIR /app

# Install PyTorch with CUDA 12.1 support FIRST (before other requirements)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Python dependencies (faiss-gpu may fail, fallback to faiss-cpu)
COPY services/ml-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || \
    (sed -i 's/faiss-gpu/faiss-cpu/g' requirements.txt && pip install --no-cache-dir -r requirements.txt)

# Install torch-geometric with CUDA support (after PyTorch)
# Using direct pip install with correct CUDA version
RUN pip install --no-cache-dir pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.5.0+cu121.html 2>/dev/null && \
    pip install --no-cache-dir torch-geometric || \
    echo "Warning: torch-geometric installation failed (GNN engines will be disabled)"

# Copy service code into /app/src
RUN mkdir -p /app/src
COPY services/ml-service/src/ /app/src/

# Copy shared modules
COPY shared /app/shared

# Define volumes for persistence
VOLUME ["/app/data/uploads"]

# Set CUDA environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=0
# GTX 1660 Ti is Turing architecture (SM 7.5)
ENV TORCH_CUDA_ARCH_LIST="7.5"

USER nemo

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8006/health || exit 1

EXPOSE 8006

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8006"]
