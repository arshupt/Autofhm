# AutoML

An automated machine learning tool that performs auto-feature engineering, model selection and hyperparameter optimisation(HPO). This project uses Genetic Algorithm to perform the model selection and HPO.

## Installation

1. `pip install Autofhm` 
2. To install from source `pip install -e .` in the project root directory(optional)
3. `pip install rich[jupyter]` If using jupyter notebook(only if using jupyter)

> NB: Second step is not required if already installed using `pip`.

## Running 

1. create a configuration file (shown below)
2. `python -m Autofhm -c path/to/config/file ` to generate a model.
   
> NB: `python -m Autofhm -h` for more options 


### Basic Usage

```python
from Autofhm import Autofhm
_temp = Autofhm(config="path/to/config/file")


# Perform feature engineering on the given dataset. Make sure the dataset path inside configuration file is valid. A Full path is preferred.   
# Feature engineered dataset will be stored inside the only in the model itself.
_temp.get_features()

# Starts the genetic algorithm to perform model selection and hyperparameter optimisations.
# refer source: Autofhm/ga/config_models.py for available models.
_temp.fit()
```

## To run tests

- Run `python test/test.py` inside the project root folder (This takes a lot of time).
- To build a specific model run `python test/test.py -c {dataset folder name}`. Where the directory is the test datasets folder.
- Results will be stored at [test/results](test/results) with file name same as to the dataset Folder name.
- The format and sample data are shown below (varies based on classification/regression)

|Date Time|accuracy|balanced_accuracy|f1|precision|recall|time to build|
|---|---|---|--|---|---|---|
|2021-06-05 10:03:09 |0.851261054498923|10.907589576871073|2.0325990974167425|22.489063492063487|0.8516438951901327|23.94397735595703|

|Date Time|r^2|mean_squared_error|mean_absolute_error|max_error|explained_variance|time to build|
|---|---|---|--|---|---|---|
|2021-06-05 10:03:09 |0.851261054498923|10.907589576871073|2.0325990974167425|22.489063492063487|0.8516438951901327|23.94397735595703|


## Config file format

1. Refer [docs/config_file_samples](docs/config_file_samples) for sample configuration file. Json file contains the default values of each variable.


## Submitting Code

1. Create a new branch with name starting like `bugfix-<issuenum>` or `feature-<featurename>` or `docs-<name>`.
2. Add the code and add also what does it do and how does it do in the comments.
3. commit the changes with the summary of the changes.
4. push to that branch and create a merge request.
