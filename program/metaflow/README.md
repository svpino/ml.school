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
MLFLOW_TRACKING_URI=http://0.0.0.0:5000
KERAS_BACKEND=jax

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






## Deployment Pipeline

The deployment pipeline supports deploying the latest model from the Model Registry to SageMaker or Azure ML. You can control the platform you want to deploy to by using the `--target` parameter.

The deployment pipeline will create a new endpoint to host the model if it doesn't exist. If the endpoint already exists, the pipeline will update the endpoint configuration with the latest model version.




### Deploying the model to SageMaker

You can run the deployment pipeline with the following command:

```bash
$ python3 deployment.py --environment=pypi run --target sagemaker --endpoint_name penguins
```

After you are done with the SageMaker endpoint, make sure you delete it to avoid unnecessary costs:

```bash
$ aws sagemaker delete-endpoint --endpoint-name penguins
```



### Deploying the model to Azure

To deploy the model to Azure, you must have an Azure subscription. If you don't have one, start by creating a [free account](https://azure.microsoft.com/en-us/pricing/purchase-options/azure-account?icid=azurefreeaccount).

Install the Azure [Command Line Interface (CLI)](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) and [configure it on your environment](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2&tabs=public). After finishing these steps, you should be able to run the following command to display your Azure subscription and configuration:

```bash
$ az account show && az configure -l
```
If it doesn't exist, create an `.env` file inside the repository main directory with the following environmemnt variables (alternatively, you can set these environment variables using the `export` command):

```bash
AZURE_SUBSCRIPTION_ID=[YOUR SUBSCRIPTION ID]
AZURE_RESOURCE_GROUP=[YOUR RESOURCE GROUP]
AZURE_WORKSPACE=[YOUR WORKSPACE]
```
Create an environment variable with the endpoint name we'll create in Azure:

```bash
$ export ENDPOINT_NAME=penguins
```

You can run the deployment pipeline using the following command:

```bash
$ python3 deployment.py --environment=pypi run --target azure --endpoint_name $ENDPOINT_NAME
```

After you are done with the Azure endpoint, make sure you delete it to avoid unnecessary costs:

```bash
$ az ml online-endpoint delete --name penguins
```
