# import libraries
import sys
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Function which loads and merges data 
    Input: the filepaths to the messages and categories data
    Output: the a merged dataframe called df of the two input datasets
    '''
    #read in messages data
    print(" reading in messages...")
    messages = pd.read_csv(messages_filepath)
    
    #read in categories data
    print(" reading in categories...")
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    print(" merging the datasets...")
    df = pd.merge(left  = messages, 
                  right = categories,
                  on = 'id',
                  how = 'left')
    
    return df

#Function to clean the data
#reads in merged dataframe from load_data function
def clean_data(df):
    '''
    Function to clean the data
    Input: merged dataframe from the load_data function
    Output: clean dataframe
    Cleaning involves:
        splitting out categories field into 36 cols
        binarising the categories
        removing duplicates
    '''
    #The categories field in df needs to be split out
    print(" splitting out categories into indivudal columns...")
    
    #Firstly create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";",expand = True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for        categories.
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #convert values in category columns to 1s or 0s
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].map(lambda x: str(x)[-1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
        #ensure categories are binary
        categories[column] = np.where(categories[column]>0,1,0)
        
    #Now we need to join this back on
    print(" concating split out categories back to df...")
    
    #First drop the original categories column from `df`
    df.drop('categories', axis = 1,inplace = True)
    
    #Then concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],  axis = 1)
    
    #Next part of cleaning is to remove any duplicates
    print(" removing duplicates...")
    df = df[~df.duplicated(keep='first')]
    
    #Return cleaned df
    return df

def save_data(df, database_filename):
    '''
    Function to save the cleaned data to a sql db
    input: dataframe to be saved (df)
           database filename
    output: saved df
    '''
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df.to_sql('clean_df2', engine, index=False) 


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