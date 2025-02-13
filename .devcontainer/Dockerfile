FROM debian:trixie-slim

RUN apt-get update && apt-get install -y  git \
    just \
    jq \
    curl \
    build-essential

# TODO: fix SSL issue and remove '--insecure' param 
RUN curl --insecure -LsSf https://astral.sh/uv/install.sh | sh

# TODO: add user to avoid working on root account

# mlflow requirement
ENV USERNAME="mlschool"
