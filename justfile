default:
    @just --list

# Display the version of the Python interpreter to ensure it is supported.
python:
    @version=$(python3 --version 2>/dev/null | awk '{print $2}'); [[ $version =~ ^3\.12\.[0-9]+$ ]] && echo "Python: $version" || echo "Python $version is not supported"

# Display the version of some of the required dependencies to ensure they are installed.
dependencies:
    @metaflow_version=$(pip3 show metaflow | grep Version | awk '{print $2}') && \
        docker_version=$(docker --version | awk '{print $3}' | sed 's/,//') && \
        jq_version=$(jq --version | awk -F'-' '{print $2}') && \
    echo "Metaflow: $metaflow_version" && \
    echo "Docker: $docker_version" && \
    echo "jq: $jq_version"


# Create file with environment variables and export them in the current shell.
env:
   @echo "KERAS_BACKEND=jax" >> .env 
   @echo "MLFLOW_TRACKING_URI=http://127.0.0.1:5000" >> .env
   @export $(cat .env | xargs)
   @cat .env



# Create and activate virtual environment
setup:
    python3 -m venv .venv
    . .venv/bin/activate
    pip3 install -U pip
    pip3 install -r requirements.txt

# Run MLflow server locally
mlflow:
    mlflow server --host 127.0.0.1 --port 5000

# Run MLflow server with SQLite backend
mlflow-sqlite:
    mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db

# Run training pipeline
train:
    python3 pipelines/training.py --environment=pypi run

# View training pipeline results
train-view:
    python3 pipelines/training.py --environment=pypi card server

# Deploy model locally
deploy-local:
    mlflow models serve -m models:/penguins/$(curl -s -X GET "$MLFLOW_TRACKING_URI/api/2.0/mlflow/registered-models/get-latest-versions" -H "Content-Type: application/json" -d '{"name": "penguins"}' | jq -r '.model_versions[0].version') -h 0.0.0.0 -p 8080 --no-conda

# Deploy model to SageMaker
deploy-sagemaker endpoint:
    mlflow sagemaker build-and-push-container
    python3 pipelines/deployment.py --environment=pypi run --target sagemaker --endpoint {{endpoint}} --region $AWS_REGION --data-capture-destination-uri s3://$BUCKET/datastore

# Deploy model to Azure
deploy-azure endpoint:
    python3 pipelines/deployment.py --environment=pypi run --target azure --endpoint {{endpoint}}

# Clean up AWS resources
cleanup-aws:
    aws cloudformation delete-stack --stack-name mlflow
    aws cloudformation delete-stack --stack-name metaflow
    aws cloudformation delete-stack --stack-name mlschool
    aws sagemaker delete-endpoint --endpoint-name $ENDPOINT_NAME

# Clean up Azure resources  
cleanup-azure:
    az ml online-endpoint delete --name $ENDPOINT_NAME --resource-group $AZURE_RESOURCE_GROUP --workspace-name $AZURE_WORKSPACE --no-wait --yes
    az group delete --name $AZURE_RESOURCE_GROUP
