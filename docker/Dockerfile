# Image for the MIRA-DB app and MCP server.
# Compose sets a different entrypoint for each service.
FROM python:3.13-slim

WORKDIR /sw

RUN apt-get update && apt-get install -y --no-install-recommends git

# Setup pip, install miradb, and clean up
RUN pip install --no-cache-dir --upgrade pip && \
    pip install "miradb[mcp] @ git+https://github.com/gyorilab/miradb.git" && \
    rm -rf /var/lib/apt/lists/*

# Set so worker and Python output appear promptly in docker logs
ENV PYTHONUNBUFFERED=1
