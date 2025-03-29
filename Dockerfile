# Use Python 3.8 as base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python package management
RUN pip install uv

# Copy the project files
COPY . .

# Create and activate virtual environment
RUN uv venv
ENV PATH="/app/.venv/bin:$PATH"

# Install the project and its dependencies
RUN uv pip install -e .

# Set environment variables with default values
ENV OLLAMA_URL=""
ENV EMBEDDING_URL=""
ENV EMBEDDING_MODEL="nomic-embed-text"
ENV POSTGRES_DB=""
ENV POSTGRES_USER=""
ENV POSTGRES_PASSWORD=""
ENV POSTGRES_HOST=""
ENV POSTGRES_PORT="5432"

# Command to run the MCP server
CMD ["uv", "--directory", "./src/parquet_mcp_server", "run", "main.py"]
