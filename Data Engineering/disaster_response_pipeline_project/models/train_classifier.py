import sys

import pandas as pd
from sqlalchemy import create_engine
import nltk
import joblib
import re
import time

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download(['punkt', 'wordnet','stopwords'])

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
	Load database and get dataset
	Input:
		database_filepath : file path of sqlite database
	Output:
		X : Features
		y : Target
        categorie_names: List of categorical columns
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('disaster_data',con=engine)

    X = df['message']
    Y = df[df.columns[5:]]
    category_names = list(Y.columns)

    return X,Y,category_names


def tokenize(text):
    """
    The function would take in a text and output the cleaned tokens.
    Input :
        text : Input text that needs to be tokenized
    Output :
        clean_tokens :cleaned up version of the toknes
    """

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Returns a model

    Output:
        cv: Grid search model object
    """

    # define the step of pipeline
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize,ngram_range=(1,2),max_df=0.75)),
    ('tfidf', TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))])

    # define the parameters to fine tuning
    parameters = {
        'vect__max_features': (None,10000,20000)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Outputs classification results

    Input:
        model  : the scikit-learn fitted model
        X_text : The X test set
        y_test : the y test classifications
        category_names : the category names

    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))

def save_model(model, model_filepath='random_forest.pkl'):
    """
    Saves the model to given path

    Input:
        model : the fitted model
        model_filepath : filepath to save model
	"""
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
