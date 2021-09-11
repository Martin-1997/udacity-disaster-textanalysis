# Table of Contents
1. [Overview](#Overview)
2. [Installation](#Installation)
3. [Files](#Files)
4. [Acknowledgements](Acknowledgements)
5. [Licence](Licence)

# Overview
This project is part of the Udacity Data Science Nanodegree Course. The goal of the project is to analyze short text messages in the scope of disaster response. Different needs like missing water or electricity should be categorized to get a better view on the disaster and to perform appropriate measurements.

# Installation

All the files were created using Jupyter Lab and the Python programming language (version 3.8.8). The following Pyhton libraries were used:
- numpy
- pandas 1.2.4
- seaborn 0.11.1
- sqlalchemy
- nltk
- re
- sklearn
- pickle
- plotly

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
