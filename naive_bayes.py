from load_data import load_sem_eval_data
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from metrics import calculate_score

def print_stance_statistics(data: pd.DataFrame) -> None:
    """
    Prints the number of tweets in each stance
    """
    stance_counts = data['Stance'].value_counts()
    print(stance_counts)

def main():
    target = 'Climate Change is a Real Concern'

    train_data, test_data = load_sem_eval_data(target)  

    #print(train_data)
    #print_stance_statistics(train_data)

    #print(test_data)
    #print_stance_statistics(test_data)

    # Split data into X and y
    X_train = train_data['Tweet']
    y_train = train_data['Stance']

    X_test = test_data['Tweet']
    y_test = test_data['Stance']

    # Vectorize data
    vectorizer = CountVectorizer(ngram_range=(4, 12)) # 4-grams to 12-grams seems to work best
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Create and train Naive Bayes Classifier
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)


    # Make predictions on test data
    y_pred = classifier.predict(X_test)

    print(calculate_score(y_test, y_pred))


if __name__ == '__main__':
    main()