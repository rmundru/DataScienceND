import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    input:
        messages_filepath : File path of the messages filepath
        categories_filepath : File path of the categories filepath


    output :
        df:Single dataframe with both the data points
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories , on = 'id')

    return df


def clean_data(df):
    """
    input:
        df:data frame of DisasterResponse
    output:
        df:data frame thats clean
    """
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = [name[ :len(name)-2] for name in row]
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype('str').str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # converting the values of realted column to 1 & 0
    categories['related'] = categories['related'].replace(2,1)

    # drop the original categories column from `df`
    df.drop(['categories'],axis=1 ,inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis = 1)

    # drop duplicates
    df.drop_duplicates(subset = 'id',inplace = True)

    return df


def save_data(df, database_filename):
    """
    input:
        df:Input dataframe with DisasterResponse cleaned data
        database_filename:name of the database file
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_data', engine, index=False)


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
