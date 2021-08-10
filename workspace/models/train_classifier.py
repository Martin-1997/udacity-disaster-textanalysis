import sys

# import libraries
from sqlalchemy import create_engine
import pandas as pd
import re
import numpy as np
# nltk
import nltk
nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# other models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# pickle
import pickle


def load_data(database_filepath):
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('TextMessages', engine)
    X = df[["message", "original", "genre"]]
    Y = df.drop(columns= ["id", "message", "original", "genre"])
    
    return X, Y, Y.columns


def tokenize(text):
    # Normalization
    
    # Convert to lower case
    text = text.lower()
    
    # Remove punctuation characters - this regex finds everything which is not a combination of letters
    # and numbers and replaces it with a whitespace
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    
    # Tokenization
    
    # Split into tokens
    words = word_tokenize(text)
    
    
    # Remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Part-of-speech tagging maybe useful here?
    # Named Entity Recognition usefuk here?
    
    # Stemming - only keep the stem of a word, simple find and replace method which removes f.e. "ing"
    # stemmed = [PorterStemmer().stem(w) for w in words]
    
    # Lemmatization - more complex appraoch using dictionaries which can f.e. map "is" and "was" to "be"
    # Lemmatize verbs by specifying pos
    lemmed_verbs = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]
    # Reduce nouns to their root form
    lemmed_nouns = [WordNetLemmatizer().lemmatize(w) for w in lemmed_verbs]
    return lemmed_nouns


def build_model():
    # pipeline = Pipeline([
    #         ('features', FeatureUnion([

    #             ('text_pipeline', Pipeline([
    #                 ('vect', CountVectorizer(tokenizer=tokenize)),
    #                 ('tfidf', TfidfTransformer())
    #             ]))
    #         ])),

    #         ('clf', MultiOutputClassifier(RandomForestClassifier()))
    #     ])

    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    return pipeline

# Calculate the average accuracy for each target column
def print_acc(name, model, y_test, y_pred):
    columns = y_test.columns
    y_pred_df = pd.DataFrame(y_pred, columns = columns)
    accuracy = (y_pred_df == y_test.reset_index().drop(["index"], axis = 1)).mean()
    print(f"Accuracy per category {name}: ")
    print(f"Average accuracy: {accuracy.mean()}")
    print(accuracy)
    return {'name' : name, 'model': model, 'accuracy' : accuracy}

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test["message"])
    result = print_acc("MultiOutputClassifier with wrapped RandomForestClassifier", model, Y_test, y_pred)

def save_model(model, model_filepath):
    # save the model to disk
    #filename = 'finalized_model.sav'
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train["message"], Y_train)
        
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