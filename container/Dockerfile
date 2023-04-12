ARG MULTI_MODEL=true

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.4-cpu AS sagemaker

RUN apt-get clean && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx

RUN pip install --upgrade pip
RUN pip install pandas
RUN pip install numpy
RUN pip install requests
RUN pip install tensorflow==2.4

FROM sagemaker AS multi-model-false
LABEL com.amazonaws.sagemaker.capabilities.multi-models=false
ENV SAGEMAKER_MULTI_MODEL=false

FROM multi-model-${MULTI_MODEL} AS final
RUN echo "multi-model enabled: ${SAGEMAKER_MULTI_MODEL}"

