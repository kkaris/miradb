# Shared base image for the MIRA-DB app and MCP server.
# Installs the miradb package, including the mcp extra.
# Runnable images are Dockerfile.app and Dockerfile.mcp.
FROM python:3.13-slim

WORKDIR /sw

RUN apt-get update && apt-get install -y --no-install-recommends git

# Setup pip, install miradb, and clean up
RUN pip install --no-cache-dir --upgrade pip && \
    pip install "miradb[mcp] @ git+https://github.com/gyorilab/miradb.git" && \
    rm -rf /var/lib/apt/lists/*
