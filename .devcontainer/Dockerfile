FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye as base

# Install a native toolchain and common build dependencies
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        binutils \
        gcc \
        g++ \
        libssl-dev \
        libffi-dev \
        libsqlite3-dev \
        sqlite3 \
        libwebkit2gtk-4.0-dev \
        curl \
        ca-certificates \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# Ensure compilers are discoverable by build tools
ENV CC=gcc
ENV CXX=g++

# Install `just`
RUN pipx install rust-just

# Install uv and increase the timeout (in seconds) for HTTP requests
RUN pipx install uv
ENV UV_HTTP_TIMEOUT=300 
ENV UV_LINK_MODE=copy


RUN if [ "$(uname -m)" = "x86_64" ]; then \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf aws/ awscliv2.zip; \
    fi

RUN if [ "$(uname -m)" = "aarch64" ]; then \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf aws/ awscliv2.zip; \
    fi

