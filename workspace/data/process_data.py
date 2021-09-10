# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads the data from CSV-files using pandas and creates a single dataframe containing all the data
    
    Parameters:
    messages_filepath (string): The filepath to the messages.csv file
    categories_filepath (string): The filepath to the categories.csv file
    
    Returns:
    Pandas Dataframe: Dataframe that contains the merged data from the csv files
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = "id")
    return df

def clean_data(df):
    '''
    Cleans the dataframe with the following operations:
    
    - expands the categories column
    - drops the child_alone-category because there is no information (all values are the same)
    - removes duplicates
    
    Parameters:
    df (Pandas Dataframe): A Dataframe created by the load_data-function
    
    Returns:
    Pandas Dataframe: The Dataframe after the cleaning operations were executed
    '''
    categories = df["categories"].str.split(pat=";", expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    category_colnames = []

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    for value in row:
        category_colnames.append(value[:-2])
        
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # We drop "child_alone" because this column contains always the same values
    categories.drop(["child_alone"], axis = 1, inplace = True)
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:] #.str[-1:]

        # convert column from string to numeric
        categories[column] =   categories[column].astype(int)
    df.drop(["categories"], axis = 1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    duplicated = df.duplicated()
    dup_sum = duplicated.sum()
    
    # drop duplicates
    df = df[~duplicated]
    
    # "related" has the values 0,1 and 2 which cannot be for a truth value. Therefore we convert all the 2's to 1's
    df["related"] = df["related"].astype(bool).astype(int)
    
    return df


def save_data(df, database_filename):
    '''
    Saves a dataframe to a SQLite database file permanently on disk
    
    Parameters: 
    df (Pandas Dataframe): The dataframe which should be saved
    database_filename (String): The name that should be given to the database file
    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('TextMessages', engine, index=False, if_exists='replace')    


def main():
    '''
    Main script function which runs all the required processing steps
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()