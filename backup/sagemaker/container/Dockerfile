FROM arm64v8/ubuntu

RUN apt-get -y update && apt-get install -y --no-install-recommends \
    build-essential wget nginx ca-certificates gcc \
    linux-headers-generic libc-dev

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh
RUN chmod a+x miniconda.sh
ENV PATH /root/miniconda3/bin/:$PATH
RUN bash miniconda.sh -b

COPY ml-dependencies.yml ./
RUN conda env create -f ml-dependencies.yml

ENV PATH /root/miniconda3/envs/ml-dependencies/bin:$PATH