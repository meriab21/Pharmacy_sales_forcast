# Rossmann-Pharmaceuticals-Sales-Prediction

## Check Dashboard:
👉 https://share.streamlit.io/depacifier/pharmacy_sales_prediction/main/app.py

**Table of Contents**

- [Rossmann-Pharmaceuticals-Sales-Prediction](#rossmann-pharmaceuticals-sales-prediction)
  - [Overview](#overview)
  - [Scenario](#scenario)
  - [Approach](#approach)
  - [Project Structure](#project-structure)
    - [data:](#data)
    - [models:](#models)
    - [notebooks:](#notebooks)
    - [scripts](#scripts)
    - [tests:](#tests)
    - [logs:](#logs)
    - [root folder](#root-folder)
  - [Installation guide](#installation-guide)

## Overview
This repository is used for week 3 challenge of 10Academy.

## Scenario
You work at Rossmann Pharmaceuticals as a Machine Learning Engineer. The finance team
wants to forecast sales in all their stores across several cities six weeks ahead of time.
Managers in individual stores rely on their years of experience as well as their personal
judgement to forecast sales.

The data team identified factors such as promotions, competition, school and state holidays,
seasonality, and locality as necessary for predicting the sales across the various stores.

Your job is to build and serve an end-to-end product that delivers this prediction to analysts
in the finance team.

## Approach
The project is divided and implemented by the following phases
- Exploration of customer purchasing behavior
- Prediction of store sales
  - Machine learning approach
  - Deep Learning approach
- Serving predictions on a web interface

## Project Structure
The repository has a number of files including python scripts, jupyter notebooks and text files. Here is their structure with a brief explanation.

### data:
- the folder where the dataset csv files are stored

### .github:
- the folder where github actions and CML workflow is integrated

### .dvc:
- the folder where dvc is managed and configured for remote data version control

### models:
- the folder where model pickle files and model reference csv files are stored

### notebooks:
- `data_exploration.ipynb`: a jupyter notebook for exploring the data
- `data_preprocessing.ipynb`: a jupyter notebook for preprocessing the data for ML and further analysis
- `lstm_forecasting.ipynb`: a jupyter notebook training an LSTM model for forecasting purpose
- `train_regression.ipynb`: a jupyter notebook training an Regression models for prediction purpose

### scripts
- `ML_modelling_utils.py`: a python script for handle model name creation and manipulation
- `data_cleaner.py`: a python script for cleaning pandas dataframes
- `data_information.py`: a python script for selecting data from a pandas dataframe
- `data_loader.py`: a python script for loading csv and excel files to a dataframe
- `data_manipulation.py`: a python script for manipulating dataframes
- `grapher.py`: a python script for plotting dataframes
- `multiapp.py`: a python script for creating a multipaged streamlit app
- `logger_creator.py`: a python script that creates python based logger and helps user log different messages
- `results_pickler.py`: a python script for collecting and pickling data
- `utlity_functions.py`: a python script for other utility functions
- `dvc_data_loader.py`: a python script for loading dvc managed source data

### tests:
- the folder containing unit tests for components in the scripts


### root folder
- `requirements.txt`: a text file lsiting the projet's dependancies
- `travis.yml`: a configuration file for Travis CI
- `app.py`: entry file for the streamlit application
- `setup.py`: a configuration file for installing the scripts as a package
- `README.md`: Markdown text with a brief explanation of the project and the repository structure.

## Installation guide
```
git clone https://github.com/DePacifier/pharmacy_sales_prediction.git
cd pharmacy_sales_prediction
pip install -r requirements.txt
```
