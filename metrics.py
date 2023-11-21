from sklearn.metrics import f1_score

def calculate_score(y_test, y_pred) -> float:
    """
    Calculates the evaluation metric for the SemEval 2016 Task 6, Subtask A
    which is the macro-average of the f1-score for "FAVOR" and the
    f1-score for "AGAINST", ignoring the "NONE" class.

    Args:
        y_test: The true labels
        y_pred: The predicted labels

    Returns:
        The calculated F1 score
    """
    mask = (y_test != 'NONE') # Remove "NONE" class tweets
    y_test_filtered = y_test[mask]
    y_pred_filtered = y_pred[mask]

    f1 = f1_score(y_test_filtered, y_pred_filtered, average='macro')
    return f1