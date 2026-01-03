# Gmail Service Docker Image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (matches existing services)
RUN useradd -m -u 1000 nemo
RUN chown -R nemo:nemo /app

# Copy requirements first for layer caching
COPY services/gmail-service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules
COPY shared /app/shared
RUN pip install -e /app/shared

# Copy service source
COPY services/gmail-service /app/gmail-service

# Set Python path
ENV PYTHONPATH=/app:/app/gmail-service/src

# Create token storage directory
RUN mkdir -p /app/tokens && chown nemo:nemo /app/tokens

# Switch to non-root user
USER nemo

# Expose port
EXPOSE 8016

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8016/health || exit 1

# Run service
CMD ["python", "-m", "uvicorn", "gmail-service.src.main:app", "--host", "0.0.0.0", "--port", "8016"]
