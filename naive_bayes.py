from load_data import load_sem_eval_data, split_data, print_stance_statistics
from metrics import calculate_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def main():
    target = 'Climate Change is a Real Concern'

    train_data, test_data = load_sem_eval_data(target)  

    #print(train_data)
    #print_stance_statistics(train_data)

    #print(test_data)
    #print_stance_statistics(test_data)

    # Split data into X and y
    X_train, y_train = split_data(train_data)
    X_test, y_test = split_data(test_data)

    # Vectorize data
    vectorizer = CountVectorizer(ngram_range=(4, 12), ) # 4-grams to 12-grams seems to work best
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