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



## Running mlflow in Ubuntu:

1. Install Docker. - https://docs.docker.com/engine/install/ubuntu/

```
$ sudo apt-get install build-essential python3.12-venv zlib1g-dev unzip
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install mlflow packaging setuptools
```
Add 5000 port to firewall.

```
$ mlflow server --host 0.0.0.0 --port 5000
```

On a different terminal:
 
```
$ export MLFLOW_TRACKING_URI=http://34.207.234.19:5000
```

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
$ aws sagemaker delete-endpoint --endpoint-name penguins
$ aws ecr delete-repository --repository-name mlflow-pyfunc --force
```



#####

Install Azure CLI: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2&tabs=public
