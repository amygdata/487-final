from load_data import load_sem_eval_data
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

def print_stance_statistics(data: pd.DataFrame):
    """
    Prints the number of tweets in each stance
    """
    stance_counts = data['Stance'].value_counts()
    print(stance_counts)

def train_naive_bayes(X_train, y_train):
    # Pipeline needed for GridSearchCV
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', MultinomialNB()),
    ])

    # Define hyperparameters to test
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'vect__max_df': [0.5, 0.75, 1.0],
        'vect__min_df': [1, 2, 3],
    }

    # Perform grid search
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # Print the best parameters
    print(grid_search.best_params_)

def main():
    target = 'Climate Change is a Real Concern'

    # Load training data
    train_data_path = 'semeval2016-task6-trainingdata.txt'
    train_data = load_sem_eval_data(train_data_path, target) 
    #print(train_data)
    #print_stance_statistics(train_data)


    # Load test data
    test_data_path = 'SemEval2016-Task6-subtaskA-testdata-gold.txt'
    test_data = load_sem_eval_data(test_data_path, target)
    #print(test_data)
    #print_stance_statistics(test_data)


    # Split data into X and y
    X_train = train_data['Tweet']
    y_train = train_data['Stance']

    X_test = test_data['Tweet']
    y_test = test_data['Stance']

    # Convert text data into numerical data
    train_naive_bayes(X_train, y_train)
    #vectorizer = CountVectorizer(ngram_range=(3, 20))
    #X_train = vectorizer.fit_transform(X_train)
    #X_test = vectorizer.transform(X_test)

    ## Create and train Naive Bayes Classifier
    #classifier = MultinomialNB()
    #classifier.fit(X_train, y_train)


    ## Make predictions on test data
    #y_pred = classifier.predict(X_test)

    ## Calculate macro-average F1 score
    #mask = (y_test != 'NONE')
    #y_test_filtered = y_test[mask]
    #y_pred_filtered = y_pred[mask]

    #f1 = f1_score(y_test_filtered, y_pred_filtered, average='macro')
    #print(f1)


if __name__ == '__main__':
    main()