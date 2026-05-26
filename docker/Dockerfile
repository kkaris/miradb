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

# Setup
RUN mkdir -p /root/.config/indra && \
    echo "[indra]" > /root/.config/indra/config.ini && \
    pip install --upgrade pip

# Install miradb
RUN pip install "miradb @ git+https://github.com/gyorilab/miradb.git@dockerize"

# Set entypoint
ENTRYPOINT ["python", "-m", "miradb.sources.app", "--port", "8003"]
