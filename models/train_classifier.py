import sys
import pandas as pd
import numpy  as np
import re
from sqlalchemy import create_engine
from sqlite3 import connect

# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''
    Function to load in data
    input: data filepath
    output:
        x: messages from the dataframe
        y: response columns from the df
        C: column names from the df
    '''
    sqlcon = connect(database_filepath)
    df = pd.read_sql('SELECT * from clean_df2', sqlcon)
    #define feature variable X
    X = df.message
    x = X.values
    
    #define target variables Y
    Y = df.iloc[:,-36:]
    y = Y.values
    
    #define column names
    C = Y.columns 
    return x, y, C


def tokenize(text):
    '''
    Function to clean and tokenise text in message field
    input raw message fld
    output tokenised and lemmatised text
    '''
    #only keep alphanumeric characters
    clean_text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    #tokenize text
    tokens = word_tokenize(clean_text)
    
    #initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    #iterate through each token
    clean_tokens = []
    for tok in tokens:
       #lemmatise:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        
        #append back
        clean_tokens.append(clean_tok)
    
    #remove stop words
    clean_tokens = [w for w in clean_tokens if w not in stopwords.words("english")]
    
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])
    
    print(pipeline.get_params())
    #params = {'clf__estimator__bootstrap' :[True,False]
    #          ,'clf__estimator__n_estimators': [5,10,20]}
    
    #cv = GridSearchCV(pipeline, param_grid=params,cv = 5)
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


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