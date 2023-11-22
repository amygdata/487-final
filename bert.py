from load_data import load_sem_eval_data, split_data, print_stance_statistics
from metrics import calculate_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optimizer
import torch

class TweetDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data


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

def get_label_tensor(labels):
    """
    Convert a list of labels to a tensor

    Input:
        - labels: list of labels
        - label_mapping: dictionary mapping label to index

    Output:
        - label_tensor: tensor of labels
    """
    label_mapping = {'FAVOR': 2, 'NONE':1, 'AGAINST': 0}

    label_nums = labels.map(label_mapping)

    return torch.tensor(label_nums.values)

def main():
    target = 'Climate Change is a Real Concern'

    train_data, test_data = load_sem_eval_data(target)  

    # Split data into X and y
    X_train, y_train = split_data(train_data)
    X_test, y_test = split_data(test_data)

    # Move tensors to GPU if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Tokenize to BERT format
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

    print(f'Total Number of Rows: {X_train.shape}')

    # Tokenize data
    X_train_ids = tokenizer.batch_encode_plus(X_train, 
                                          padding=True,       
                                          return_tensors='pt')   

    X_test = tokenizer.batch_encode_plus(X_test, 
                                          padding=True,       
                                          return_tensors='pt')   

    # Convert labels to tensors
    y_train = get_label_tensor(y_train)
    y_test = get_label_tensor(y_test)

    train_data = TensorDataset(X_train_ids['input_ids'], X_train_ids['attention_mask'], y_train)
    train_loader = DataLoader(train_data, batch_size=16)

    test_data = TensorDataset(X_test['input_ids'], X_test['attention_mask'], y_test)
    test_loader = DataLoader(test_data, batch_size=16)

    # Create BERT model
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # Fine Tune BERT model
    optimizer = get_optimizer(model, lr=5e-5, weight_decay=0)
    epochs = 25
    num_itr = 0

    model.to('cuda')

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:            
            num_itr += 1
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch No. {0}--Iteration No. {1}-- batch loss = {2:.4f}'.format(
            epoch + 1,
            num_itr,
            loss.item()
            ))





if __name__ == '__main__':
    main()