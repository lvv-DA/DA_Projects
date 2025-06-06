# Base image - Python 3.9 slim (Debian Bookworm)
FROM python:3.9-slim-bookworm

# Set environment variables for build and runtime
# PYTHONUNBUFFERED=1: Ensures Python output (logs) is immediately sent to stdout/stderr, crucial for Docker logs.
# PIP_NO_CACHE_DIR=1: Prevents pip from storing downloaded packages, reducing image size.
# STREAMLIT_SERVER_HEADLESS=true: Tells Streamlit not to open a browser window, as it's running in a headless environment.
# STREAMLIT_SERVER_FILE_WATCHER_TYPE=none: Disables file watching, important for production as files won't change live.
# STREAMLIT_BROWSER_GATHER_USAGE_STATS=false: Disables Streamlit's usage data collection for privacy.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install minimal system dependencies
# libgomp1: Required for OpenMP, often used by ML libraries like scikit-learn, XGBoost.
# libgl1: Provides OpenGL libraries, sometimes needed by visualization tools (e.g., Plotly, Matplotlib).
# curl: For the healthcheck endpoint.
# && rm -rf /var/lib/apt/lists/*: Cleans up apt cache to keep image size small.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1 \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements.txt first to leverage Docker's build cache.
# If requirements.txt doesn't change, this layer and subsequent RUN (pip install) will be cached.
COPY requirements.txt .

# Install Python packages.
# --upgrade pip: Ensures you have the latest pip.
# --prefer-binary: Tries to install pre-compiled wheels, faster and avoids build dependencies.
# -r requirements.txt: Installs all listed dependencies.
RUN pip install --upgrade pip && \
    pip install --prefer-binary -r requirements.txt

# Copy the rest of the application files.
# .dockerignore should be set up to exclude unnecessary files (e.g., venv/, __pycache__/).
COPY . .

# Health check (for production deployments).
# Ensures the Streamlit server is actually responsive by hitting its dedicated health endpoint.
# --interval: How often to check.
# --timeout: How long to wait for a response.
# --retries: How many times to retry before marking as unhealthy.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose Streamlit port.
# This informs Docker that the container listens on this port.
EXPOSE 8501

# Define the command to run when the container starts.
# ENTRYPOINT is preferred for the main process as it makes the container an executable.
# --server.port=8501: Ensures Streamlit listens on this port inside the container.
# --server.address=0.0.0.0: Makes Streamlit accessible from outside the container (crucial for host mapping).
ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0"]