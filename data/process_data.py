# import libraries

import pandas as pd
from sqlalchemy import create_engine
import sys
import warnings
warnings.filterwarnings("ignore")


def load_data(messages_filepath, categories_filepath):
    """loads the specified message and category data
    Args:
        messages_filepath (string): The file path of the messages csv
        categories_filepath (string): The file path of the categories cv
    Returns:
        df (pandas dataframe): The combined messages and categories df
    """
    messages = pd.read_csv(messages_filepath)
    categories_n = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories_n,how='inner',on='id')

    return df   


def clean_data(df):
    """Cleans the data:
        - drops duplicates
        - removes messages missing classes
        - cleans up the categories column
    Args:
        df (pandas dataframe): combined categories and messages df
    Returns:
        df (pandas dataframe): Cleaned dataframe with split categories
    """

    # expand the catagories columns
    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[[0]]

    # get all the categories name
    category_colnames = [i[:-2] for i in row.values.tolist()[0]]
    categories.columns = category_colnames

    for column in categories.columns:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-',expand=True)[1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop original category column
    df.drop(['categories'],axis=1,inplace=True)

    # add individual category columns
    df = pd.concat([df,categories],axis=1)
    df.dropna(subset=category_colnames, inplace=True)
    
    # remove duplicate and clean it
    df = df.drop_duplicates(subset='message')
    df['related'].replace({2:0},inplace=True)
    return df
    


def save_data(df, database_filename):
    ## create the sqlite engine and save the dataset
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False,if_exists='replace')  


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