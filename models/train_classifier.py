import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_data(database_filepath):
    '''
    load_data
    Load data from SQLite database

    database_filepath: filepath to the database

    return:
    X: dataframe with messages
    y: dataframe with type of disasters (labels)
    category_name: list of type of disasters
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response', engine)
    X = df['message']
    y = df.drop(['id', 'message', 'genre'], axis=1)
    category_name = y.columns
    return X, y, category_name


def tokenize(text):
    '''
    tokenize
    Perform some pre-processing steps with text value

    text: column contains messages

    return:
    words: valuable keyword of messages
    '''
    #Split text into words using NLTK
    words = word_tokenize(text.lower())
    #Remove punctuation marks
    words = [word for word in words if word.isalnum()]
    #Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    return words


def build_model():
    '''
    build_model
    Build a machine learning model to categorize the type of disasters

    return:
    cv: the final model to be applied
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
    'clf__estimator__criterion': ["gini", "entropy"]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model
    Evaluate the model the define if it's suitable to be applied

    model: the built model 
    X_test: the test features data
    Y_test: the test labels data
    category_names: list of the types of disasters
    '''
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=category_names)
      
    print("Labels:", category_names)
    
    print("Classification Report:\n")
    for col in category_names:
        print('- Column: ', col)
        print(classification_report(Y_test[col], Y_pred[col]))
    
    print("Accuracy:")
    for col in category_names:
        print('- Column: ', col)
        print(accuracy_score(Y_test[col], Y_pred[col]))
    pass


def save_model(model, model_filepath):
    '''
    save_model
    Save model to apply for categorizing type of disaster with the unseen messages

    model: the built model
    model_filepath: filepath to save the model to
    '''
    file = open(model_filepath, 'wb')
    pickle.dump(model, file)
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