import pandas as pd
import chardet 

def detect_encoding(file_path: str):
    """
    Detects the encoding of a file
    """
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())

    return result['encoding']

def load_sem_eval_data(txt_path: str, target: str):
    """
    Loads the SemEval 2016 dataset and extracts rows that contain the target

    Args:
        txt_path: Path to the SemEval 2016 dataset (.txt format)
        target: The target to extract from the dataset

    Returns:
        Pandas dataframe containing the dataset
    """
    # Load dataset
    data = pd.read_csv(txt_path, sep='\t', encoding=detect_encoding(txt_path))

    # Remove targets we do not care about
    data = data.loc[data['Target'] == target]

    # Remove '#SemSt' from the 'Tweet' column
    data['Tweet'] = data['Tweet'].str.replace('#SemST', '')

    # Make tweets lowercase
    data['Tweet'] = data['Tweet'].str.lower()

    return data