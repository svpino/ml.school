# Cleaning Up AWS Resources

The command below removes the `mlflow` CloudFormation stack that we created to run an MLflow server in a cloud instance:

```shell
aws cloudformation delete-stack --stack-name mlflow
```

The command below removes the `metaflow` CloudFormation stack that we created to run the pipelines in a remote compute cluster:

```shell
aws cloudformation delete-stack --stack-name metaflow
```

You can run the following command to delete the endpoint from Sagemaker:

```shell
just sagemaker-delete
```

Finally, when you finish using AWS, you can delete the stack and all the related resources by running the following command:

```shell
just aws-teardown
```

