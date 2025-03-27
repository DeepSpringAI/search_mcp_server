# Use an official Python runtime as a parent image (Debian-based)
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Clone the repository into the container
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Create and activate a virtual environment
RUN uv venv

# Set the virtual environment's Python as the default interpreter for the system
ENV PATH="/app/.venv/bin:$PATH"

# Install the package using uv
RUN uv pip install -e .

# Copy the .env file into the container
COPY .env /app/.env

# Use /bin/sh to run commands and ensure compatibility with your requirements
CMD ["uv", "--directory", "/app/src/parquet_mcp_server", "run", "main.py"]
