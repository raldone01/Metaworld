# A dockerfile for building Metaworld environment images for testing and CI
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get -y update \
    && apt-get install --no-install-recommends -y \
    libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev \
    xvfb unzip patchelf ffmpeg cmake swig git \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY . /usr/local/metaworld/
WORKDIR /usr/local/metaworld/

RUN uv sync --no-default-groups --group testing --compile-bytecode --managed-python --no-python-downloads --python /usr/local/bin/python3

ENTRYPOINT ["/usr/local/metaworld/docker/entrypoint"]
