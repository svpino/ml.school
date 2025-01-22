# Cleaning Up AWS Resources

When you finish using your AWS account, clean up everything to prevent unnecessary charges.

The command below removes the `mlflow` CloudFormation stack that we created to run an MLflow server in a cloud instance:

```shell
aws cloudformation delete-stack --stack-name mlflow
```

The command below removes the `metaflow` CloudFormation stack that we created to run the pipelines in a remote compute cluster:

```shell
aws cloudformation delete-stack --stack-name metaflow
```

If you aren't planning to return to the program, you can also remove the CloudFormation stack configuring your account and permissions. Keep in mind that the resources created by this stack do not cost money, so you can keep them around indefinitely if you want:

```shell
aws cloudformation delete-stack --stack-name mlschool
```

Finally, you can run the following command to delete the endpoint from Sagemaker:

```shell
just sagemaker-delete
```