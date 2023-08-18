# Week 2: Experiment tracking

## ML experiment: Process of building an ML model

### Experiment tracking
Process of keeping track of all the relevant information from a ML experiment, which includes:

Source code
Environment
Data
Model
Hyperparameters
Metrics

#### Why is experiment tracking important?

Reproducibility
Organization
Optimization

### MLflow

"An open source platform for the machine learning lifecycle". It is a python package that contains the following modules:

Tracking
Models
Model registry
Projects

Mlflow tracking module allows you to organize your experiments into runs, and keep track of:

Parameters
Metrics
Metadata
Artifacts
Models

MLflow automatically logs extra information about the run:

Source code
Version of the code (git commit)
Start and end time
Author

Experiment run: Each trial in an ML experiment
Run artifact: Any file that is associated with an ML run
Experiment metadata

## Install and run MLflow

They recommended to use an environment

'''
conda create --name env_name
'''

'''
conda activate env_name
'''

Install the requirements

'''
pip install -r requirements.txt
'''

## Running an experiment on MLflow

hyperopt.github.io/hyperopt/getting-started/search-spaces/