# AutoML

An automated machine learning tool that performs auto-feature engineering, model selection and hyperparameter optimisation. This project uses Genetic Algorithm to perform the later two.

## Installation

1. `pip install Autofhm` 
2. To install from source `pip install -e .` in the project root directory
3. `pip install rich[jupyter]` If using jupyter notebook(optional)


## To run tests

- Run `python test/all.py` inside the project root folder.
- To build a specific model run `python test/{dataset}/test.py`. Where the directory is the test datasets folder.


## Config file format

1. Refer [docs/config_file_samples](docs/config_file_samples) for sample confgruation file. Json file contains the default values of each variable.


## Submitting Code

1. Create a new branch with name starting like `bugfix-<issuenum>` or `feature-<featurename>` or `docs-<name>`.
2. Add the code and add also what does it do and how does it do in comments.
3. commit the changes with the summmary of the changes.
4. push to that branch and create a merge request.
