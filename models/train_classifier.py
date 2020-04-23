import sys

import joblib
import nltk
import pandas as pd
from sqlalchemy import create_engine
import string

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

import warnings
warnings.filterwarnings("ignore")

stop_words = set(stopwords.words('english'))
porter = PorterStemmer()


def load_data(database_filepath):
    """Loads X and Y and gets category names
    Args:
        database_filepath (str): string filepath of the sqlite database
    Returns:
        X (pandas dataframe): Feature data, just the messages
        Y (pandas dataframe): Classification labels
        category_names (list): List of the category names for classification
    """

    engine = create_engine('sqlite:///' + database_filepath )
    df = pd.read_sql(database_filepath.split('/')[-1][:-3], engine)
    X = df['message']
    y = df.iloc[:,4:]
    categories_name = y.columns.tolist()

    return X,y, categories_name


def tokenize(text):
    """Basic tokenizer that do lower case, removes punctuation, numbers and stopwords then lemmatizes
    Args:
        text (string): input message to tokenize
    Returns:
        tokens (list): list of cleaned tokens in the message
    """
    
    # making text to lower case
    text = text.lower()
    
    # remove numeric characters
    result = re.sub(r'\d+', '', text)

    # tokenize the text
    tokens = word_tokenize(result)

    # Remove punctuations
    rm_pun= [word for word in tokens if word.isalnum()]

    ## Remove stop words
    tokens = [w for w in rm_pun if not w in stop_words]

    # Stemming the text
    clean_tokens = [porter.stem(t) for t in tokens]
    
    return clean_tokens


def build_model():

    """Returns the GridSearchCV object to be used as the model
    Args:
        None
    Returns:
        cv (scikit-learn GridSearchCV): Grid search model object
    """

    clf = AdaBoostClassifier()
    pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
                    ('clf', MultiOutputClassifier(clf))
                        ])

    return pipeline



def evaluate_model(model, X_test, Y_test, category_names):
    """Prints multi-output classification results
    Args:
        model (pandas dataframe): the scikit-learn fitted model
        X_text (pandas dataframe): The X test set
        Y_test (pandas dataframe): the Y test classifications
        category_names (list): the category names
    Returns:
        None
    """

    Y_pred = model.predict(X_test)

    # Print out the full classification report
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """dumps the model to the given filepath
    Args:
        model (scikit-learn model): The fitted model
        model_filepath (string): the filepath to save the model to
    Returns:
        None
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