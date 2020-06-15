import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load dataframe from filepaths
    INPUT
    messages_filepath -- str, link to file
    categories_filepath -- str, link to file
    OUTPUT
    df - pandas DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    """Clean data included in the DataFrame and transform categories part
    INPUT
    df -- type pandas DataFrame
    OUTPUT
    df -- cleaned pandas DataFrame
    """
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.loc[0]
    colnames = []
    for entry in row:
        colnames.append(entry[:-2])
    category_colnames = colnames
    print('Column names:', category_colnames)
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype(int)
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    # Removing entry that is non-binary
    df = df[df['related'] != 2]
    print('Duplicates remaining:', df.duplicated().sum())
    return df
    
def save_data(df, database_filename):
    """Saves DataFrame (df) to database path"""
    name = 'sqlite:///' + database_filename
    engine = create_engine(name)
    df.to_sql('Disasters', engine, index=False)


def main():
    """Runs main functions: Loads the data, cleans it and saves it in a database"""
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