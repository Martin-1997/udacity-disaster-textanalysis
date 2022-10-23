# Table of Contents
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Installation](#installation)
- [Files](#files)
- [Acknowledgements](#acknowledgements)
- [Licence](#licence)

# Overview
This project is part of the Udacity Data Science Nanodegree Course. The goal of the project is to analyze short text messages in the scope of disaster response. Different needs like missing water or electricity should be categorized to get a better view on the disaster and to perform appropriate measurements.

# Installation

All the files were created using VS Code and the Python programming language (version 3.10.8). To recreate the environment, follow the following steps:

- Install Python 3.10.8
- Clone the repository
- Create a virtual environment with venv: `python3 -m venv venv`
- Load the virtual environment with `source venv/bin/activate`
- Inside the main directory, run `pip3 install -r requirements.txt`

You need to follow the following steps to process the data, create the model and run the webapp:

- Go to the workspace directory `cd workspace`
- Run the folloing command to run the ETL pipeline that cleans data and stores in database:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- Run the following command to run the ML pipeline that trains classifier and saves:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
- Go to the app directory `cd app`
- Run the following command to start the web app:
        `python run.py`
- Go to http://0.0.0.0:3001/

The current code does not train the model with the whole dataset which allows to test the code in a more time efficient way. To run the code with the full dataset, open workspace/models/train_classifier.py and set full_dataset to True
# Files

- categories.csv and messages.csv: Raw data used to build the model
- ./workspace: environment from the Udacity workspace IDE. Here are all the scripts which need to be executed to automatically create and train the model based on the raw data
- ETL Pipeline Preparation.ipynb: Jupyter notebook file which contains step-by-step descriptions and code to convert the raw data into a database file
- ML Pipeline Preparation.ipynb: Jupyter notebook file which contains step-by-step descriptions and code to use the data from the database file to train a model
- DisasterData.db: database file containing the transformed data from categories.csv and messages.csv

# Acknowledgements

I want to express gratitudes for the team at Udacity which provides a great course on the topic of Data Science.

# Licence

The code is available to anybody for personal and commercial use with the GNU General Public License v3.0. The used libraries may have there own different licences.
I do not guarantee for any correctness or timelines of the results. No warranty is provided.
