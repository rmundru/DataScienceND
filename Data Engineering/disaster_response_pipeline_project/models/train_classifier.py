import sys
import nltk
import re
nltk.download(['punkt', 'wordnet'])

import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report, accuracy_score



def load_data(database_filepath):
    """
    The following function would read the data from the database tables
    Input:
        database_filepath:Database file filepath
    Output:
        X:Message value that we want to predict the categories
        Y:True values in binary format
        category_names: Names of the categories
    """

    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('disaster_data',con = engine)
    X = df['message'].values
    Y = df[df.columns[4:]].values
    category_names = list(df.columns[4:])
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
    This function defines a model that would be used to predict the categories
    using the pipeline and girdsearch

    Output:
        cv: model to be using in training and predicting
    """
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer = tokenizer)),
    ('tfidf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier(n_estimator = 50)))
    ])

    parameters = {'clf__estimator__max_features':['sqrt':0.5],
                  'clf__estimator__n_estimators':[50,100,150]}
    cv = GridSearchCV(pipeline,parameters,cv=5,n_jobs =10)

    return cv



def evaluate_model(model, X_test, Y_test, category_names):

    """
    This function evaluates the model by passing in the training and test database
    Input:
        model : Model to be used for training and testing
        X_test: test messages
        Y_test: test values for categories
        category_names : Names of the categories

    """

    #predict on X_test
    Y_pred = model.predict(X_test)

    #build classification report on every column

    print(classification_report(Y_test,Y_pred,target_name = category_names))

    for i in range(Y_test.shape[1]):
        print('%25 accuracy :%.2f' %(category_names[i],accuracy_score(Y_test[:,i],Y_pred[:,i])))




def save_model(model, model_filepath):
    """
    The following function will save the model to a given a filepath
    Input :
        model : model used to train and predicting
        model_filepath :path to save the trained file
    """
    joblib.dump(model,model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()
a
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
