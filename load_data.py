import pandas as pd
import chardet 

def detect_encoding(file_path: str) -> str:
    """
    Detects the encoding of a file
    """
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())

    return result['encoding']

def preprocess_data(data: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Preprocesses the data

    Args:
        data: The dataframe to preprocess
        target: The target to extract from the dataset

    Returns:
        The preprocessed dataframe
    """
    # Deep copy the dataframe
    data = data.copy()

    # Remove targets we do not care about
    data = data.loc[data['Target'] == target]

    # Remove '#SemSt' from the 'Tweet' column
    data['Tweet'] = data['Tweet'].str.replace('#SemST', '')

    # Make tweets lowercase
    data['Tweet'] = data['Tweet'].str.lower()

    return data

def load_sem_eval_data(target: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Loads the SemEval 2016 dataset and extracts rows that contain the target

    Args:
        target: The target (e.g., "Climate Change is a Real Concern") to extract from the dataset

    Returns:
        Two pandas dataframes containing the training and test datasets
    """

    # Load training dataset
    train_data_path = 'semeval2016-task6-trainingdata.txt'
    train_data = pd.read_csv(train_data_path, sep='\t', encoding=detect_encoding(train_data_path))

    # Load test dataset
    test_data_path = 'SemEval2016-Task6-subtaskA-testdata-gold.txt'
    test_data = pd.read_csv(test_data_path, sep='\t', encoding=detect_encoding(test_data_path))

    # Preprocess training and test data
    train_data = preprocess_data(train_data, target)
    test_data = preprocess_data(test_data, target)

    return train_data, test_data