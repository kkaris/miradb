FROM ubuntu:24.04

ENV DIRPATH=/sw
WORKDIR $DIRPATH

RUN apt-get update && \
    apt-get install -y git python3 python3-pip python3.12-venv && \
    ln -s /usr/bin/python3 /usr/bin/python

# Create and activate virtual environment (required for ubuntu 24.04)
ENV VENV_PATH=$DIRPATH/venv
RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Setup pip and install miradb
RUN pip install --upgrade pip && \
    pip install "miradb @ git+https://github.com/gyorilab/miradb.git"

# Set so worker and Python output appear promptly in docker logs
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["gunicorn", "-w", "2", "-t", "300", "-b", "0.0.0.0:5000", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info", "--capture-output", "miradb.sources.app:app"]
