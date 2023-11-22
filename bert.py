from load_data import load_sem_eval_data, split_data, print_stance_statistics
from metrics import calculate_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optimizer
import torch

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

def get_device():
    """
    Return the device you will use for training/testing.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_loader(X, y, tokenizer, batch_size):
    """
    Creates the data loader for SemEval data

    Args:
        X: The input data
        y: The labels
        batch_size: The batch size
    """
    X_ids = tokenizer.batch_encode_plus(X,
                                        padding=True,
                                        return_tensors='pt')

    y_tensors = get_label_tensor(y)

    data = TensorDataset(X_ids['input_ids'], X_ids['attention_mask'], y_tensors)

    return DataLoader(data, batch_size=batch_size)

def main():
    target = 'Climate Change is a Real Concern'

    train_data, test_data = load_sem_eval_data(target)  

    # Split data into X and y
    X_train, y_train = split_data(train_data)
    X_test, y_test = split_data(test_data)

    # Tokenize to BERT format
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

    # Create data loaders
    train_loader = create_loader(X_train, y_train, tokenizer, batch_size=16)
    test_loader = create_loader(X_test, y_test, tokenizer, batch_size=16)

    # Create BERT model
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # Fine Tune BERT model
    optimizer = get_optimizer(model, lr=5e-5, weight_decay=0)
    epochs = 10
    num_itr = 0

    device = get_device()
    model.to(device)

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

    # Evaluate the model
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    for batch in test_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get the predicted labels
        _, preds = torch.max(outputs.logits, dim=1)
        
        # Count the number of correct predictions, ignore the "NONE" class
        mask = (labels != 1)
        correct_predictions += torch.sum(preds[mask] == labels[mask])
        total_predictions += torch.sum(mask)

    # Calculate the accuracy
    print(correct_predictions, total_predictions)
    accuracy = correct_predictions.double() / total_predictions.double()

    print('Test Accuracy: {:.4f}'.format(accuracy))





if __name__ == '__main__':
    main()