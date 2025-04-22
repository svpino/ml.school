set dotenv-load
set positional-arguments

KERAS_BACKEND := env("KERAS_BACKEND", "tensorflow")
MLFLOW_TRACKING_URI := env("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
ENDPOINT_NAME := env("ENDPOINT_NAME", "penguins")
BUCKET := env("BUCKET", "")
AWS_REGION := env("AWS_REGION", "us-east-1")
AWS_ROLE := env("AWS_ROLE", "")

default:
    @just --list

# Run project unit tests
test:
    uv run -- pytest

# Display version of required dependencies
[group('setup')]
@dependencies:
    uv_version=$(uv --version) && \
        just_version=$(just --version) && \
        docker_version=$(docker --version | awk '{print $3}' | sed 's/,//') && \
        jq_version=$(jq --version | awk -F'-' '{print $2}') && \
    echo "uv: $uv_version" && \
    echo "just: $just_version" && \
    echo "docker: $docker_version" && \
    echo "jq: $jq_version"

# Run MLflow server
[group('setup')]
@mlflow:
    uv run -- mlflow server --host 127.0.0.1 --port 5000

# Set up required environment variables
[group('setup')]
@env:
    if [ ! -f .env ]; then echo "KERAS_BACKEND={{KERAS_BACKEND}}\nMLFLOW_TRACKING_URI={{MLFLOW_TRACKING_URI}}" >> .env; fi
    cat .env


# Run training pipeline
[group('training')]
@train:
    uv run -- python pipelines/training.py \
        --with retry \
        --environment conda run

# Run training pipeline card server 
[group('training')]
@train-viewer:
    uv run -- python pipelines/training.py \
        --environment conda card server

# Serve latest registered model locally
[group('serving')]
@serve:
    uv run -- mlflow models serve \
        -m models:/penguins/$(curl -s -X GET "{{MLFLOW_TRACKING_URI}}/api/2.0/mlflow/registered-models/get-latest-versions" \
        -H "Content-Type: application/json" -d '{"name": "penguins"}' \
        | jq -r '.model_versions[0].version') -h 0.0.0.0 -p 8080 --no-conda

# Invoke local running model with sample request
[group('serving')]
@invoke:
    uv run -- curl -X POST http://0.0.0.0:8080/invocations \
        -H "Content-Type: application/json" \
        -d '{"inputs": [{"island": "Biscoe", "culmen_length_mm": 48.6, "culmen_depth_mm": 16.0, "flipper_length_mm": 230.0, "body_mass_g": 5800.0, "sex": "MALE" }]}'

# Display number of records in SQLite database
[group('serving')]
@sqlite:
    uv run -- sqlite3 penguins.db "SELECT COUNT(*) FROM data;"

# Generate fake traffic to local running model
[group('monitoring')]
@traffic:
    uv run -- python pipelines/traffic.py \
        --environment conda run \
        --samples 200

# Generate fake labels in SQLite database
[group('monitoring')]
@labels:
    uv run -- python pipelines/labels.py \
        --environment conda run

# Run the monitoring pipeline
[group('monitoring')]
@monitor:
    uv run -- python pipelines/monitoring.py \
        --config config config/local.json \
        --environment conda run

# Run monitoring pipeline card server 
[group('monitoring')]
@monitor-viewer:
    uv run -- python pipelines/monitoring.py \
        --environment conda card server \
        --port 8334


# Set up your AWS account using and configure your local environment.
[group('aws')]
@aws-setup user region='us-east-1':
    uv run -- python scripts/aws.py setup \
        --stack-name mlschool \
        --region {{region}} \
        --user {{user}}

# Delete the CloudFormation stack and clean up AWS configuration.
[group('aws')]
@aws-teardown region='us-east-1':
    uv run -- python scripts/aws.py teardown \
        --stack-name mlschool \
        --region {{region}}


# Deploy MLflow Cloud Formation stack
[group('aws')]
@aws-mlflow:
    aws cloudformation create-stack \
        --stack-name mlflow \
        --template-body file://cloud-formation/mlflow-cfn.yaml

# Create mlschool.pem file
[group('aws')]
@aws-pem:
    aws ssm get-parameters \
        --names "/ec2/keypair/$(aws cloudformation describe-stacks \
            --stack-name mlflow \
            --query "Stacks[0].Outputs[?OutputKey=='KeyPair'].OutputValue" \
            --output text)" \
        --with-decryption | python3 -c 'import json;import sys;o = json.load(sys.stdin);print(o["Parameters"][0]["Value"])' > mlschool.pem

    chmod 400 mlschool.pem

# Connect to the MLflow remote server
[group('aws')]
@aws-remote:
    ssh -i "mlschool.pem" ubuntu@$(aws cloudformation \
        describe-stacks --stack-name mlflow \
        --query "Stacks[0].Outputs[?OutputKey=='PublicDNS'].OutputValue" \
        --output text)

# Deploy model to Sagemaker
[group('aws')]
@sagemaker-deploy:
    uv run -- python pipelines/deployment.py \
        --config config config/sagemaker.json \
        --environment conda run \
        --backend backend.Sagemaker

# Invoke Sagemaker endpoint with sample request
[group('aws')]
@sagemaker-invoke:
    uv run -- awscurl --service sagemaker --region "$AWS_REGION" \
        $(aws sts assume-role --role-arn "$AWS_ROLE" \
            --role-session-name mlschool-session \
            --profile "$AWS_USERNAME" --query "Credentials" \
            --output json | \
            jq -r '"--access_key \(.AccessKeyId) --secret_key \(.SecretAccessKey) --session_token \(.SessionToken)"' \
        ) -X POST -H "Content-Type: application/json" \
        -d '{"inputs": [{"island": "Biscoe", "culmen_length_mm": 48.6, "culmen_depth_mm": 16.0, "flipper_length_mm": 230.0, "body_mass_g": 5800.0, "sex": "MALE" }] }' \
        https://runtime.sagemaker."$AWS_REGION".amazonaws.com/endpoints/"$ENDPOINT_NAME"/invocations


# Delete Sagemaker endpoint
[group('aws')]
@sagemaker-delete:
    aws sagemaker delete-endpoint --endpoint-name "$ENDPOINT_NAME"

# Generate fake traffic to Sagemaker endpoint
[group('aws')]
@sagemaker-traffic:
    uv run -- python pipelines/traffic.py \
        --config config config/sagemaker.json \
        --environment conda run \
        --backend backend.Sagemaker \
        --samples 200

# Generate fake labels in SQLite database
[group('aws')]
@sagemaker-labels:
    uv run -- python pipelines/labels.py \
        --config config config/sagemaker.json \
        --environment conda run \
        --backend backend.Sagemaker

# Run monitoring pipeline card server
[group('aws')]
@sagemaker-monitor-viewer:
    uv run -- python pipelines/monitoring.py \
        --environment conda card server \

# Run the monitoring pipeline
[group('aws')]
@sagemaker-monitor:
    uv run -- python pipelines/monitoring.py \
        --config config config/sagemaker.json \
        --environment conda run \
        --backend backend.Sagemaker

# Run training pipeline in AWS
[group('aws')]
@aws-train:
    METAFLOW_PROFILE=production uv run -- python pipelines/training.py \
        --environment conda run \
        --with batch \
        --with retry

# Create a state machine for the training pipeline in AWS Step Functions
[group('aws')]
@aws-train-sfn-create:
    METAFLOW_PROFILE=production uv run -- python pipelines/training.py \
        --environment conda step-functions create

# Trigger the training pipeline in AWS Step Functions
[group('aws')]
@aws-train-sfn-trigger:
    METAFLOW_PROFILE=production uv run -- python pipelines/trainining.py \
        --environment conda step-functions trigger

# Deploy model to Sagemaker
[group('aws')]
@aws-deploy:
    METAFLOW_PROFILE=production uv run -- python pipelines/deployment.py \
        --config-value config '{"target": "{{ENDPOINT_NAME}}", "data-capture-uri": "s3://{{BUCKET}}/datastore", "ground-truth-uri": "s3://{{BUCKET}}/ground-truth", "region": "{{AWS_REGION}}", "assume-role": "{{AWS_ROLE}}"}' \
        --environment conda run \
        --backend backend.Sagemaker \
        --with batch

# Create a state machine for the deployment pipeline in AWS Step Functions
[group('aws')]
@aws-deploy-sfn-create:
    METAFLOW_PROFILE=production uv run -- python pipelines/deployment.py \
        --config-value config '{"target": "{{ENDPOINT_NAME}}", "data-capture-uri": "s3://{{BUCKET}}/datastore", "ground-truth-uri": "s3://{{BUCKET}}/ground-truth", "region": "{{AWS_REGION}}", "assume-role": "{{AWS_ROLE}}"}' \
        --environment conda step-functions create

# Trigger the deployment pipeline in AWS Step Functions
[group('aws')]
@aws-deploy-sfn-trigger:
    METAFLOW_PROFILE=production uv run -- python pipelines/deployment.py \
        --environment conda step-functions trigger \
        --backend backend.Sagemaker


