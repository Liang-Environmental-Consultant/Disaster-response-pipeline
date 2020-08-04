import sys
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

import pickle

def load_data(database_filepath):
    """
    Load data from database
 
    Output: X, Y, category_names
    
    """
    table_name = 'Disaster_Response_Table'
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name, engine)  
    X = df['message']
    Y = df.drop(['message', 'id', 'original', 'genre'], axis=1)
    
    category_names= Y.columns
    return X, Y, category_names

def tokenize(text):
     """
     Normalize, tokenize and lemmatize text string
    
    Input:
    text: string containing disaster message for processing
       
    Returns:
    List that containing normalized and lemmatize word tokens
    """
    # Detect URLs
     url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
     detected_urls = re.findall(url_regex, text)
     for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Normalize and tokenize and remove punctuation
     tokens = nltk.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    
    # Remove stopwords
     tokens = [t for t in tokens if t not in stopwords.words('english')]

    # Lemmatize
     lemmatizer=WordNetLemmatizer()
     tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
     return tokens

def build_model():
    """
    Build machine learning model (KNeighborsClassifier)
    Input: 
    clean-tokens: X_train, Y_train, X_test, Y_test 
    Returns:
    pipline: sklearn.model_selection.GridSearchCV. 
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])
    
#   parameters = {    
#       'clf__estimator__n_neighbors': [1,10]
#      }
    
#    cv = GridSearchCV(pipeline, param_grid=parameters)
    model = pipeline
    
    return model
    
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performance
    Input:
    model: sklearn.model_selection.GridSearchCV.  It contains a sklearn estimator.
    X_test: disaster messages.
    Y_test: disaster categorie for each message
    category_names: disaster message categories
    Output: 
    classification report of each category
    """
    Y_pred=model.predict(X_test)
    i = 0
    while i<len(Y_test.columns):
         print(Y_test.columns[i])
         print(classification_report(Y_test[Y_test.columns[i]], Y_pred[i]))
         i = i+1
  
def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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
        
#        print('Evaluating model...')
#        evaluate_model(model, X_test, Y_test, category_names)

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