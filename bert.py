from load_data import load_sem_eval_data, split_data, print_stance_statistics
from metrics import calculate_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
import torch.optim as optimizer

def get_optimizer(net, lr, weight_decay):
    """
    FROM HOMEWORK 3
    Return the optimizer (Adam) you will use to train the model.

    Input:
        - net: model
        - lr: initial learning_rate
        - weight_decay: weight_decay in optimizer
    """

    return optimizer.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)

def main():
    target = 'Climate Change is a Real Concern'

    train_data, test_data = load_sem_eval_data(target)  

    # Split data into X and y
    X_train, y_train = split_data(train_data)
    X_test, y_test = split_data(test_data)


    # Tokenize to BERT format
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

    # Tokenize data
    X_train = tokenizer.batch_encode_plus(X_train, 
                                          padding=True,       
                                          return_tensors='pt')   

    X_test = tokenizer.batch_encode_plus(X_test, 
                                          padding=True,       
                                          return_tensors='pt')   

    print(X_train)

    train_loader = DataLoader(X_train, batch_size=16)
    test_loader = DataLoader(X_test, batch_size=16)

    # Create BERT model
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # Fine Tune BERT model
    optimizer = get_optimizer(model, 0.001, 0.0001)



if __name__ == '__main__':
    main()