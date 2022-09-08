# import libraries

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sqlalchemy import create_engine

import sys
import pickle

import pandas as pd

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def load_data(database_filepath):
    """
    Loads data from Database

    Parameter:
    database_filepath - File path to Database

    Output:
    X - Learning parameters
    Y - Label parameter
    Y.columns - Columns of Label parameter
    """

    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM Data', engine)

    # Divide data in learning and label parameter
    X = df[['message']]
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    return X, Y, Y.columns


def tokenize(text):
    """
    Tokenizes and lemmatizes text.

    Parameter:
    text - Text to tokenize and lemmatize.

    Output:
    tokens - Clean tokens of text
    """

    # Replace URL with 'urlplaceholder', normalise case
    text = re.sub(r'http\S+', 'urlplaceholder', text)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # get stop words, initilize Lemmatizer
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # lemmatize, tokenize text
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    Initilize pipeline of classifier

    Output:
    pipeline - Pipeline that can be trained
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('model', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the performance of the model, prints evaluation

    Parameter:
    model - Model to evaluate
    X_test - Test data of learning parameters
    Y_test - Test data of label parameter
    category_names - Names of the categories
    """

    # predict test data
    y_pred = model.predict(X_test.message)

    # calculate accuracy
    accuracy = (y_pred == Y_test).mean()
    print(accuracy)

    # print evaluation
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))


def save_model(model, model_filepath):
    """
    Saves model in model_filepath
    """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Loads data, builds model, trains model, evaluates model, saves model
    """

    if len(sys.argv) == 3:
        # get passed file paths
        database_filepath, model_filepath = sys.argv[1:]

        # load data
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        # split data in trainings and test data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        # build model
        print('Building model...')
        model = build_model()

        # train model
        print('Training model...')
        model.fit(X_train.message, Y_train)

        # evaluate model
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        # save model
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
