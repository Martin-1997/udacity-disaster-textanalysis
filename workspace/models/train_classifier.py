import sys

# import libraries
from sqlalchemy import create_engine
import pandas as pd
import re
import numpy as np
# nltk
import nltk
# The nltk packages should be better installed systemwide as described here:
# https://www.nltk.org/data.html
# nltk.download('stopwords')
# nltk.download('wordnet') # download for lemmatization
# nltk.download('punkt')
# nltk.download('omw-1.4')
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
from sklearn.metrics import classification_report
# pickle
import pickle

full_dataset = False

def load_data(database_filepath):
    '''
    Loads a SQLite database from disk, formats the data and returns the X and Y columns used for training a prediction model
    
    Parameters:
    database_filepath (String): Filepath of the database on disk
    
    Returns:
    Pandas Dataframe: A Dataframe that contains the columns to predict on
    Pandas Dataframe: A Dataframe that contains the columns which should be predicted
    List<String>: A list of the column names that should be predicted
    '''
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('TextMessages', engine)
    X = df[["message", "original", "genre"]]
    Y = df.drop(columns= ["id", "message", "original", "genre"])
    
    # Only the first 100 to speed up debugging
    if full_dataset:
        return X, Y, Y.columns
    else:
        return X[:100], Y[:100], Y.columns


def tokenize(text):
    '''
    Functions that performs the following operations on the text:
    - converts to lowercase
    - removes puntuation
    - tokenizes text into word tokens
    - removes stopwords (english languages)
    - lemmatizes all the verbs
    - lemmatizes all the nouns
    
    Parameters:
    text (String): The text which should be processed
    
    Returns:
    List<String>: List of word tokens
    '''
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
    '''
    This function creates a predefined model pipeline object and returns it
    
    Returns:
    Pipeline: Predefined model pipeline
    '''
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
    # Check for available parameters to optimize
    # pipe_parameters = pipeline.get_params().keys()
        
    model_parameters = {
        # vect
        # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

        # tfidf
        # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
        'tfidf__norm' : ['l1','l2'],
        # 'tfidf__use_idf' : [True, False],
        #'tfidf__smooth_idf': [True, False],
        # 'tfidf__sublinear_tf' : [True, False],

        # clf
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        'clf__estimator__criterion' : ['gini', 'entropy'],
        'clf__estimator__n_estimators': [100], # , 200],
    }

    model = GridSearchCV(pipeline, param_grid=model_parameters) 
    
    return model

def print_acc(name, model, y_test, y_pred):
    '''
    Calculate the average f1 score, precision and recall for each target column and prints it in the console.
    Returns also a dictionary with the model, its name and classification report (contains f1 score, recall and precision)
    
    Parameters:
    name (String): Name for the model
    model (Model): Model (necessary because in the end a dictionary containing the model gets returned)
    y_test (Pandas Dataframe): True values for the y-column(s)
    y_pred (Pandas Dataframe): Predicted values for the y-column(s)
    
    Returns:
    dict: Dictionary containing the model name(key: name), the model(key: model) and the classification report of the model (key:report)
    '''
    ## old code
    #     columns = y_test.columns
    #     y_pred_df = pd.DataFrame(y_pred, columns = columns)
    #     accuracy = (y_pred_df == y_test.reset_index().drop(["index"], axis = 1)).mean()
    #     print(f"Accuracy per category {name}: ")
    #     print(f"Average accuracy: {accuracy.mean()}")
    #     print(accuracy)
    #     return {'name' : name, 'model': model, 'accuracy' : accuracy}
    
    columns = y_test.columns
    y_pred_df = pd.DataFrame(y_pred, columns = columns)
    accuracy = (y_pred_df == y_test.reset_index().drop(["index"], axis = 1)).mean()
    report = classification_report(y_true = y_test,
                              y_pred = y_pred,
                              target_names = list(y_test.columns),
                            #  output_dict = True,
                              zero_division = 0)
    print(f"F1 score, recall and precision per category {name}: ")
    print(report)
    return {'name' : name, 'model': model, 'report' : report}

def evaluate_model(model, X_test, Y_test):
    '''
    Create predictions for the test data based on a trained model. The evaluated f1 score, precision and recall is going to be printed on the screen.
    
    Parameters:
    model (Model): The model used to make the predictions who was earlier trained on the training data
    X_test (Pandas Dataframe): The data used for testing the model
    Y_test (Pandas Dataframe): The true results for the column(s) which should be predicted. Needed for evaluating the model after making predictions.
    '''
    y_pred = model.predict(X_test["message"])
    result = print_acc("MultiOutputClassifier with wrapped RandomForestClassifier", model, Y_test, y_pred)
    
def create_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred)

def save_model(model, model_filepath):
    '''
    Stores a model permanently to disk.
    
    Parameters:
    model (Model): The model which should be stored to disk
    model_filepath(String): Path containing the filename with file extension, where the model should be saved
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    Main functions serving as a entry point for the program
    '''
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
        evaluate_model(model, X_test, Y_test)

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