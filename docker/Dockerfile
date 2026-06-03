FROM python:3.13-slim

WORKDIR /sw

RUN apt-get update && apt-get install -y --no-install-recommends git

# Setup pip, install miradb, and clean up
RUN pip install --no-cache-dir --upgrade pip && \
    pip install "miradb @ git+https://github.com/gyorilab/miradb.git" && \
    rm -rf /var/lib/apt/lists/*

# Set so worker and Python output appear promptly in docker logs
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["gunicorn", "-w", "2", "-t", "300", "-b", "0.0.0.0:5000", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info", "--capture-output", "miradb.sources.app:app"]
