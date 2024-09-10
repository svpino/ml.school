TODO:
    * Monitoring pipeline
    * Tuning pipeline
    * (training) Running card server
    * (monitoring) Running card server


```
python3 training.py --environment=pypi card server
```


Ubuntu:

```
$ sudo apt-get update
$ sudo apt-get install build-essential python3.12-venv zlib1g-dev unzip

DOCKER
$ sudo usermod -a -G docker $USER
$ newgrp docker
```

Windows: Need WSL


```
$ mlflow deployments create -t sagemaker --name penguins -m models:/penguins/1 -C region_name=us-east-1 -C instance-type=ml.m4.xlarge -C instance-count=1
```




## Preparing your local environment

1. Fork the program's [GitHub Repository](https://github.com/svpino/ml.school) and clone it on your local computer.

2. Create and activate a virtual environment. The code in the repository was written using Python 3.12, so make sure you use the same Python version.

```bash
$ python3 -m venv .venv 
$ source .venv/bin/activate
```

3. Once the virtual environment is active, you can update `pip` and install the libraries in the `requirements.txt` file:

```bash
$ pip3 install -U pip
$ pip3 install -r requirements.txt
```

4. Install Docker. You'll find [installation instructions](https://docs.docker.com/engine/install/) on their site for your particular environment. After you install it, you can verify Docker is running using the following command:

```bash
$ docker ps
```

## Installing MLflow

MLflow is a platform-agnostic machine learning lifecycle management tool that will allow us to track experiments and share and deploy models. 

1. The `mlflow` library is part of the `requirements.txt` file that we installed previously, so you don't need to install it separately if you are planning to run MLflow on your local computer. If you want to run MLflow on a different environment, you can install it using the following command:

```bash
$ pip3 install mlflow
```

2. Once installed, you can run the MLflow server and navigate to [`http://127.0.0.1:5000`](http://127.0.0.1:5000) in your web browser to see the user interface:

```bash
$ mlflow server --host 127.0.0.1 --port 5000
```

By default, MLflow tracks experiments and stores data in files inside a local `./mlruns` directory. You can change the location of the tracking directory or use a SQLite database to store the tracking data using the parameter `--backend-store-uri`:

```bash
$ mlflow server --host 127.0.0.1 --port 5000 \
    --backend-store-uri sqlite:///mlflow.db
```

For more information on the MLflow server, run the following command:

```bash
$ mlflow server --help
```

## Training Pipeline

The training pipeline trains, evaluates, and registers a model in the MLflow model registry.

1. Create an `.env` file inside the repository's main directory with the environment variables below. Make sure to replace `[MLFLOW URI]` with the tracking URI of your MLflow server. If you are running MLflow on your local computer, you don't need to specify the tracking URI, or you can set it to `http://127.0.0.1:5000`.

```bash
MLFLOW_TRACKING_URI=[MLFLOW URI]
KERAS_BACKEND=jax
```

Notice that we are setting the Keras backend to `jax`. This will allows Keras tu use JAX as the backend for training the model. You can also try different backends, for example, `tensorflow` or `torch`.

2. Export the environment variables from the `.env` file in your current shell:

```bash
$ export $(cat .env | xargs)
```

### Running the training pipeline

Run the pipeline using the following command:

```bash
$ python3 training.py --environment=pypi run
```

After the pipeline finishes, you should see a new version of the model in the MLflow model registry.

For more information on the training pipeline and the parameters you can use to customize it, you can run the following command:

```bash
$ python3 training.py --environment=pypi run --help
```


## Deploying the Model

You can deploy the model to a variety of deployment targets. Except when deploying the model locally, you'll run the feployment pipeline specifying the target platform using the `--target` parameter. This pipeline will connect to the platform, create a new endpoint to host the model, and run a few samples to test that everything is working as expected.

For specific information on each supported deployment target, follow the respective links below:

* [Deploying the model as a local inference server](#deploying-the-model-as-a-local-inference-server)
* [Deploying the model to SageMaker](#deploying-the-model-to-sagemaker)
* [Deploying the model to Azure Machine Learning](#deploying-the-model-to-azure-machine-learning)

### Deploying the model as a local inference server

To deploy your model locally, you can use the `mflow models serve` command and specify the model version you want to deploy from the model registry. This command will use the active Python environment to execute the model.

This command starts a local server listening on the specified port and network interface. Make sure you replace `[MODEL VERSION]` with the version of the model you want to deploy:

```bash
$ mlflow models serve -m models:/penguins/[MODEL VERSION] \
    -h 0.0.0.0 -p 8080 \
    --no-conda
```

You can now test the model by sending a request to the server. The following command should return a prediction for the provided input:

```bash
$ curl -X POST http://0.0.0.0:8080/invocations \
    -H "Content-Type: application/json" \
    -d '{"inputs": [{
            "island": "Biscoe",
            "culmen_length_mm": 48.6,
            "culmen_depth_mm": 16.0,
            "flipper_length_mm": 230.0,
            "body_mass_g": 5800.0,
            "sex": "MALE"
        }]}'
```

You can find more information about local deployments in [Deploy MLflow Model as a Local Inference Server](https://mlflow.org/docs/latest/deployment/deploy-model-locally.html).

### Deploying the model to SageMaker

1. [Create a new AWS account](https://aws.amazon.com/free/) if you don't have one.

2. To deploy the model in SageMaker, we'll need access to `ml.m4.xlarge` instances. By default, the quota for most new accounts is zero, so you'll need to request a quota increase. You can do this in your AWS account under *Service Quotas* > *AWS Services* > *Amazon SageMaker*. Find `ml.m4.xlarge for endpoint usage` and request a quota increase of 8 instances.

3. Go to the *IAM service* and create a new role with the name `penguins`. We'll use this role to define the permissions we need to run the deployment pipeline.

4. After creating the role, modify its trust relationship to allow any user to assume this role. Open the *Trust relationships* tab and modify its content using the document below. Make sure you replace `[AWS ACCOUNT ID]` with your AWS account ID:

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

5. Open the *Permissions* tab and create an inline policy for the `penguins` role using the document below. This policy will allow any user assuming the role to access the required resources to deploy the model in SageMaker:

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

6. Under the *IAM service*, create a new user account. We'll use this user to interact with the platform through the console and the command line interface.

7. Open the *Permissions* tab and create an inline policy for the user using the document below. This policy will allow the user to assume the `penguins` role. Make sure to replace `[AWS ACCOUNT ID]` with your AWS account ID.

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

8. Open the *Security Credentials* tab and create an *access key*. Write down the **Access Key ID** and **Secret Access Key** values so you can use them later.

9. [Install the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) on your environment.

10. Configure the AWS CLI using the following command. You will need to specify your **Access Key**, **Secret Access Key**, and the **region** where you'll be deploying the model. Make sure to replace `[AWS USERNAME]` with the name of the user you created before:

```bash
$ aws configure --profile [AWS USERNAME]
```

11. We need to configure the command line interface to use the `penguins` role by defining a profile for the role in the `~/.aws/config` file. Open the file and add the lines below. Make sure to replace `[AWS ACCOUNT ID]`, `[AWS USERNAME]`, and `[AWS REGION]` with the appropriate values. After this, you should be able to take advantage of the role's permissions at the command line by using the `--profile` option on every AWS command. 

```bash
[profile penguins]
role_arn = arn:aws:iam::[AWS ACCOUNT ID]:role/penguins
source_profile = [AWS USERNAME]
region = [AWS REGION]
```

12. Export the `AWS_PROFILE` environment variable for the current session. This will allow you to take advantage of the `penguins` role's permissions on every command without having to specify the `--profile` option:

```bash
$ export AWS_PROFILE=penguins
```

13. If it doesn't exist, create an `.env` file inside the repository's main directory with the environment variables below. Make sure to replace `[MLFLOW URI]` and `[AWS REGION]` with the appropriate values:

```bash
MLFLOW_TRACKING_URI=[MLFLOW URI]
ENDPOINT_NAME=penguins

SAGEMAKER_REGION=[AWS REGION]
```

14. Export the environment variables from the `.env` file in your current shell:

```bash
$ export $(cat .env | xargs)
```

15. To host the model in SageMaker, we need to build a Docker image and push it to the Elastic Container Registry (ECR) in AWS. You can accomplish this by running the following command:

```bash
$ mlflow sagemaker build-and-push-container
```

#### Running the deployment pipeline

After you finish setting up your AWS account, you can run the deployment pipeline from the repository's main directory using the following command:

```bash
$ python3 deployment.py --environment=pypi \
    run --target sagemaker \
    --endpoint $ENDPOINT_NAME
```

As soon as you are done with the SageMaker endpoint, make sure you delete it to avoid unnecessary costs:

```bash
$ aws sagemaker delete-endpoint --endpoint-name $ENDPOINT_NAME
```



### Deploying the model to Azure Machine Learning

1. Create a [free Azure account](https://azure.microsoft.com/en-us/pricing/purchase-options/azure-account?icid=azurefreeaccount) if you don't have one.

2. Install the Azure [Command Line Interface (CLI)](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) and [configure it on your environment](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2&tabs=public). After finishing these steps, you should be able to run the following command to display your Azure subscription and configuration:

```bash
$ az account show && az configure -l
```

3. In the Azure Portal, find the *Resource providers* tab under your subscription. Register the `Microsoft.Cdn` and the `Microsoft.PolicyInsights` providers.

4. To deploy the model to an endpoint, we need to request a quota increase for the virtual machine we'll be using. In the Azure Portal, open the *Quotas* tab and filter the list by the *Machine learning* provider, your subscription, and your region. Request a quota increase for the `Standard DSv2 Family Cluster Dedicated vCPUs`. Set the new quota limit to 16.

5. If it doesn't exist, create an `.env` file inside the repository's main directory with the environment variables below. Make sure to replace `[MLFLOW URI]` and `[AZURE SUBSCRIPTION ID]` with the appropriate values:

```bash
MLFLOW_TRACKING_URI=[MLFLOW URI]
ENDPOINT_NAME=penguins

AZURE_SUBSCRIPTION_ID=[AZURE SUBSCRIPTION ID]
AZURE_RESOURCE_GROUP=mlschool
AZURE_WORKSPACE=main

```

6. Export the environment variables from the `.env` file in your current shell:

```bash
$ export $(cat .env | xargs)
```

#### Running the deployment pipeline

After you finish setting up your Azure account, you can run the deployment pipeline from the repository's main directory using the following command:

```bash
$ python3 deployment.py --environment=pypi \
    run --target azure \
    --endpoint $ENDPOINT_NAME
```

For more information on the deployment pipeline and the parameters you can use to customize it, you can run the following command:

```bash
$ python3 deployment.py --environment=pypi run --help
```

After you are done with the Azure endpoint, make sure you delete it to avoid unnecessary costs:

```bash
$ az ml online-endpoint delete --name $ENDPOINT_NAME \
    --resource-group $AZURE_RESOURCE_GROUP \
    --workspace-name $AZURE_WORKSPACE \
    --yes
```

You can also delete the entire resource group if you aren't planning to use it anymore. This will delete all the resources you created to host the model:

```bash
$ az group delete --name $AZURE_RESOURCE_GROUP
```