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

python3 week2/preprocess_data.py --raw_data_path data/ --dest_path output/

We are going to load, pre process, tune hyperparameters and then register a model in MLflow

### Pre processing

1. Import libraries
```python
import os
import pickle
import click
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
```
2. create a function to preprocess and export the data
```python
def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv
```

3. Import datasets. We are using the NYC yellow taxi data for January, February and March 2022 from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page in parquet format

```python
df_train = read_dataframe(os.path.join(raw_data_path, f"yellow_tripdata_2022-01.parquet"))
df_val = read_dataframe(os.path.join(raw_data_path, f"yellow_tripdata_2022-02.parquet"))
df_test = read_dataframe(os.path.join(raw_data_path, f"yellow_tripdata_2022-03.parquet"))
```
4. Extract the target values, the tip amount
```python
    target = 'tip_amount'
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values
```
5. Fit the DictVectorizer and preprocess data
```python  
    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_val, dv, fit_dv=False)
    X_test, _ = preprocess(df_test, dv, fit_dv=False)
```
6. Create dest_path folder unless it already exists 
```python  
    dest_path = 'output'    
    os.makedirs(dest_path, exist_ok=True)
```
7. Save DictVectorizer and datasets
```python 
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))
```
### Training

1. Import libraries
```python 
import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
```
2. Define functions to read the .pkl
```python 
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
```
3. set mlflow
```bash
mlflow ui
```

```python 
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.sklearn.autolog()
```
4. train model and track in mlflow
 ```python 
 with mlflow.start_run():

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
```

### hyperparameter tunning  of the RandomForestRegressor using optuna

In this step, you'll need to launch a tracking server. This way we will also have access to the model registry.

```bash
mlflow server --backend-store-uri sqlite:///mydb.sqlite
```
If you have problems, try to uninstall and install again alembic

```bash
pip3 uninstall alembic
pip3 install alembic
```

1. Import libraries
```python 
import os
import pickle
import click
import mlflow
import optuna

from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("random-forest-hyperopt")
```
2. Import datasets, define objective function with hyperparameters to tune and run optuna
```python 
X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 50, 1),
        'max_depth': trial.suggest_int('max_depth', 1, 20, 1),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, 1),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4, 1),
        'random_state': 42,
        'n_jobs': -1
    }

with mlflow.start_run():
    mlflow.log_params(params)
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mlflow.log_metric('rmse', rmse)

    print(rmse)

sampler = TPESampler(seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=10) 
```

### Register model

1. Import libraries
```python
import os
import pickle
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
```

2. Set Mlflow
```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()
```

3. define experiment names and hyperparameters
```python
HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state', 'n_jobs']
```

4. define functions to read the .pkl and train the model
```python
def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):

    with mlflow.start_run():
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
        X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

        for param in RF_PARAMS:
            params[param] = int(params[param])

        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        # Evaluate model on the validation and test sets
        val_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)
```

5. register model
```python
client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n, # top n models to evaluate
        order_by=["metrics.test_rmse ASC"]
    )[0]

    # Register the best model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, name="rf-best-model")
```