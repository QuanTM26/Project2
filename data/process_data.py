import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    Load data from the csv files and merge to a single pandas dataframe
    
    messages_filepath: filepath to message csv file
    categories_filepath: filepath to categories csv file

    return:
    df: a dataset merging message and categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, on="id")
    return df


def clean_data(df):
    '''
    clean_data
    Perform some pre-processing steps to clean the data 

    df: a dataset merging message and categories

    return:
    df: concatenate the original dataframe with the new 'categories' dataframe
    '''
    categories = pd.read_csv('data/disaster_categories.csv')
    
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df = pd.DataFrame.drop_duplicates(df)
    
    #drop null values
    df = df.drop('original', axis=1)
    df = df.dropna(how='any')

    #drop invalid value in 'related' column 
    df = df[(df['related']==1) | (df['related']==0)]

    return df

def save_data(df, database_filename):
    '''
    save_data
    Save the data in an SQLite Database
    
    df: the final dataframe after cleaning steps
    database_filename: database path to save data to
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_response', engine, index=False, if_exists="replace")
    pass  


def main():
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