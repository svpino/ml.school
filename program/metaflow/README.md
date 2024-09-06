```
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -U pip
$ pip install -r requirements.txt
```


```
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

```
python3 training.py --environment=pypi run

python3 training.py --environment=pypi card server
```


```
mlflow ui
```

## Running everything in Ubuntu:

```
$ sudo apt-get update
$ sudo apt-get install build-essential python3.12-venv zlib1g-dev unzip
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

Add 5000 port to firewall.

$ mlflow server --host 0.0.0.0 --port 5000

Create a .env file with the following environment variables:

```bash
MLFLOW_TRACKING_URI=http://0.0.0.0:5000
KERAS_BACKEND=jax
```

Run training pipeline:
$ python3 training.py --environment=pypi run


1. Install Docker. - https://docs.docker.com/engine/install/ubuntu/
```
$ sudo usermod -a -G docker $USER
$ newgrp docker
```

```
$ source .venv/bin/activate
```
    
    
To serve the model, I first need to install the requirements.txt file.

```
$ mlflow models serve -m models:/penguins/1 -p 8080 --no-conda
```

DEPLOYING

```
$ mlflow sagemaker build-and-push-container
$ mlflow deployments create -t sagemaker --name penguins -m models:/penguins/1 -C region_name=us-east-1 -C instance-type=ml.m4.xlarge -C instance-count=1
```


Delete the ECR repository when done:

```

$ aws ecr delete-repository --repository-name mlflow-pyfunc --force


```



#####

Install Azure CLI: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2&tabs=public






## Deploying the Model

The deployment pipeline supports deploying the latest model from the model registry to SageMaker or Azure ML. You can control the platform you want to deploy to by using the `--target` parameter.

The deployment pipeline will create a new endpoint to host the model if it doesn't exist. If the endpoint already exists, the pipeline will update the endpoint configuration with the latest model version.




### Deploying the model to SageMaker

1. [Create a new AWS account](https://aws.amazon.com/free/) if you don't have one.

2. Go to the *IAM service* and create a new role with the name `penguins`. We'll use this role to define the permissions we need to run the deployment pipeline.

3. After creating the role, modify its trust relationship to allow any user to assume this role. Open the *Trust relationships* tab and modify its content using the document below. Make sure you replace `[AWS ACCOUNT ID]` with your AWS account ID:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::[AWS ACCOUNT ID]:root"
            },
            "Action": "sts:AssumeRole",
            "Condition": {}
        }
    ]
}
```

4. Open the *Permissions* tab and create an inline policy for the `penguins` role using the document below. This policy will allow any user assuming the role to access the required resources to deploy the model in SageMaker:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "IAM",
            "Effect": "Allow",
            "Action": [
                "iam:PassRole",
                "iam:GetRole"
            ],
            "Resource": "*"
        },
        {
            "Sid": "SageMaker",
            "Effect": "Allow",
            "Action": [
                "sagemaker:ListEndpoints",
                "sagemaker:DescribeEndpoint",
                "sagemaker:CreateEndpoint",
                "sagemaker:UpdateEndpoint",
                "sagemaker:DescribeEndpointConfig",
                "sagemaker:CreateEndpointConfig",
                "sagemaker:DescribeModel",
                "sagemaker:CreateModel",
                "sagemaker:DeleteEndpoint",
                "sagemaker:ListTags",
                "sagemaker:AddTags",
                "sagemaker:InvokeEndpoint"
            ],
            "Resource": "*"
        },
        {
            "Sid": "ECR",
            "Effect": "Allow",
            "Action": [
                "ecr:DescribeRepositories",
                "ecr:CreateRepository",
                "ecr:GetAuthorizationToken",
                "ecr:InitiateLayerUpload",
                "ecr:UploadLayerPart",
                "ecr:CompleteLayerUpload",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
                "ecr:DeleteRepository",
                "ecr:PutImage"
            ],
            "Resource": "*"
        },
        {
            "Sid": "S3",
            "Effect": "Allow",
            "Action": [
                "s3:CreateBucket",
                "s3:ListBucket",
                "s3:ListAllMyBuckets",
                "s3:GetBucketLocation",
                "s3:PutObject",
                "s3:PutObjectTagging",
                "s3:GetObject",
                "s3:DeleteObject"
            ],
            "Resource": "arn:aws:s3:::*"
        }
    ]
}
```

5. Under the *IAM service*, create a new user account. We'll use this user to interact with the platform through the console and the command line interface.

6. Open the *Permissions* tab and create an inline policy for the user using the document below. This policy will allow the user to assume the `penguins` role. Make sure to replace `[AWS ACCOUNT ID]` with your AWS account ID.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "sts:AssumeRole",
            "Resource": "arn:aws:iam::[AWS ACCOUNT ID]:role/penguins"
        }
    ]
}
```

7. Open the *Security Credentials* tab and create an *access key*. Write down the **Access Key ID** and **Secret Access Key** values so you can use them later.

8. [Install the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) on your environment.

9. Configure the AWS CLI using the following command. You will need to specify your **Access Key**, **Secret Access Key**, and the **region** where you'll be deploying the model. Make sure to replace `[AWS USERNAME]` with the name of the user you created before:

```bash
$ aws configure --profile [AWS USERNAME]
```

10. We need to configure the command line interface to use the `penguins` role by defining a profile for the role in the `~/.aws/config` file. Open the file and add the lines below. Make sure to replace `[AWS ACCOUNT ID]`, `[AWS USERNAME]`, and `[AWS REGION]` with the appropriate values. After this, you should be able to take advantage of the role's permissions at the command line by using the `--profile` option on every AWS command. 

```bash
[profile penguins]
role_arn = arn:aws:iam::[AWS ACCOUNT ID]:role/penguins
source_profile = [AWS USERNAME]
region = [AWS REGION]
```

11. Export the `AWS_PROFILE` environment variable for the current session. This will allow you to take advantage of the `penguins` role's permissions on every command without having to specify the `--profile` option:

```bash
$ export AWS_PROFILE=penguins
```

12. If it doesn't exist, create an `.env` file inside the repository's main directory with the environment variables below. Make sure to replace `[MLFLOW URI]` and `[AWS REGION]` with the appropriate values:

```bash
MLFLOW_TRACKING_URI=[MLFLOW URI]
ENDPOINT_NAME=penguins

SAGEMAKER_REGION=[AWS REGION]
```

13. Export the environment variables from the `.env` file in your current shell:

```bash
$ export $(cat .env | xargs)
```

After this, you can run the deployment pipeline with the following command:

```bash
$ python3 deployment.py --environment=pypi run --target sagemaker --endpoint $ENDPOINT_NAME
```

**Important**: As soon as you are done with the SageMaker endpoint, make sure you delete it to avoid unnecessary costs:

```bash
$ aws sagemaker delete-endpoint --endpoint-name $ENDPOINT_NAME
```



### Deploying the model to Azure

To deploy the model to Azure, you must have an Azure subscription. If you don't have one, start by creating a [free account](https://azure.microsoft.com/en-us/pricing/purchase-options/azure-account?icid=azurefreeaccount).

Install the Azure [Command Line Interface (CLI)](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) and [configure it on your environment](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2&tabs=public). After finishing these steps, you should be able to run the following command to display your Azure subscription and configuration:

```bash
$ az account show && az configure -l
```
If it doesn't exist, create an `.env` file inside the repository main directory with the following environmemnt variables (alternatively, you can set these environment variables using the `export` command):

```bash
MLFLOW_TRACKING_URI=[MLFLOW URI]
ENDPOINT_NAME=penguins

AZURE_SUBSCRIPTION_ID=[YOUR SUBSCRIPTION ID]
AZURE_RESOURCE_GROUP=[YOUR RESOURCE GROUP]
AZURE_WORKSPACE=[YOUR WORKSPACE]
```
Set the environment variables from the `.env` file in your current shell:

```bash
$ export $(cat .env | xargs)
```
You can now run the deployment pipeline using the following command:

```bash
$ python3 deployment.py --environment=pypi run --target azure --endpoint $ENDPOINT_NAME
```

After you are done with the Azure endpoint, make sure you delete it to avoid unnecessary costs:

```bash
$ az ml online-endpoint delete --name $ENDPOINT_NAME --resource-group $AZURE_RESOURCE_GROUP --workspace-name $AZURE_WORKSPACE
```