#!/bin/bash
# This script upgrades the packages a SageMaker Studio Kernel Application

set -eux

pip install -q --upgrade pip
pip install -q --upgrade awscli boto3
pip install -q --upgrade scikit-learn==0.23.2
pip install -q --upgrade PyYAML==6.0
pip install -q --upgrade sagemaker
