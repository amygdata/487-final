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

    # Create n_grams
    vectorizer = CountVectorizer(ngram_range=(4, 12)) # 4-grams to 12-grams seems to work best
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Create and train Naive Bayes Classifier
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)


    # Make predictions on test data
    y_pred = classifier.predict(X_test)

    # Calculate macro-average F1 score
    mask = (y_test != 'NONE')
    y_test_filtered = y_test[mask]
    y_pred_filtered = y_pred[mask]

    f1 = f1_score(y_test_filtered, y_pred_filtered, average='macro')
    print(f1)


if __name__ == '__main__':
    main()