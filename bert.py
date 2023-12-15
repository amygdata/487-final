from load_data import load_sem_eval_data, split_data, print_stance_statistics
from metrics import calculate_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optimizer
import torch
import pandas as pd

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

def define_loss_function(weights):
    """
    Return loss fuction. Use class weights to fix lopsided training data
    """
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    return nn.CrossEntropyLoss(weight = class_weights)

def create_loader(X: pd.DataFrame, y: pd.DataFrame, tokenizer, batch_size: int) -> DataLoader:
    """
    Creates the data loader for SemEval data

    Args:
        X: The input data, a list of tweets
        y: The labels, a list of stances
        batch_size: The batch size
    """
    X_ids = tokenizer.batch_encode_plus(X,
                                        padding=True,
                                        return_tensors='pt')

    y_tensors = get_label_tensor(y)

    data = TensorDataset(X_ids['input_ids'], X_ids['attention_mask'], y_tensors)

    return DataLoader(data, batch_size=batch_size)

def train_model(model, train_loader, optimizer, device, loss_function, num_epochs):
    num_itr = 0

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            num_itr += 1
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_function(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch No. {0}--Iteration No. {1}-- batch loss = {2:.4f}'.format(
            epoch + 1,
            num_itr,
            loss.item()
            ))

    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Get the predicted labels
            _, preds = torch.max(outputs.logits, dim=1)

            # i think don't do this -> Count the number of correct predictions, ignore the "NONE" class
            #mask = (labels != 1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += torch.sum(preds)
            preds = preds
            labels = labels

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate precision, recall, and F1 score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)

    # Calculate average F1 score
    avg_f1 = sum(f1) / len(f1)

    # Print or return the results
    print('Precision (favor): {:.4f}'.format(precision[0]))
    print('Recall (favor): {:.4f}'.format(recall[0]))
    print('F1 Score (favor): {:.4f}'.format(f1[0]))

    print('Precision (neutral): {:.4f}'.format(precision[1]))
    print('Recall (neutral): {:.4f}'.format(recall[1]))
    print('F1 Score (neutral): {:.4f}'.format(f1[1]))

    print('Precision (against): {:.4f}'.format(precision[2]))
    print('Recall (against): {:.4f}'.format(recall[2]))
    print('F1 Score (against): {:.4f}'.format(f1[2]))

    print('Average F1 Score: {:.4f}'.format(avg_f1))

    # Calculate the accuracy
    print(correct_predictions, total_predictions)
    accuracy = correct_predictions.double() / total_predictions.double()

    print('Test Accuracy: {:.4f}'.format(accuracy))

def load_reddit_data():
    reddit_body_path = '/content/drive/MyDrive/reddit_body_data.txt'
    body_test = pd.read_csv(reddit_body_path, sep='\t', encoding=detect_encoding(reddit_body_path))

    # Load test dataset
    reddit_title_path = '/content/drive/MyDrive/reddit_title_data.txt'
    title_test = pd.read_csv(reddit_title_path, sep='\t', encoding=detect_encoding(reddit_title_path))

    # Preprocess training and test data
    body_data = body_test.copy()
    title_data = title_test.copy()

    body_data['Text'] = body_data['Text'].str.lower()
    title_data['Text'] = title_data['Text'].str.lower()
    body_data['Text'] = body_data['Text'].fillna('')
    title_data['Text'] = title_data['Text'].fillna('')

    return body_data, title_data

def evaluate_reddit(tokenizer, model, device):
    body_data, title_data = load_reddit_data()
    x_body, y_body = split_reddit(body_data)
    x_title, y_title = split_reddit(title_data)
    body_loader = create_loader(x_body, y_body, tokenizer, batch_size=16)
    title_loader = create_loader(x_title, y_title, tokenizer, batch_size=16)
    model.eval()
    all_preds = []
    all_labels = []
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in body_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            print(input_ids.shape)
            print(attention_mask.shape)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Get the predicted labels
            _, preds = torch.max(outputs.logits, dim=1)

            # i think don't do this -> Count the number of correct predictions, ignore the "NONE" class
            #mask = (labels != 1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += torch.sum(preds)
            preds = preds
            labels = labels

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate precision, recall, and F1 score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)

    # Calculate average F1 score
    avg_f1 = sum(f1) / len(f1)

    # Print or return the results
    print('Precision (favor): {:.4f}'.format(precision[0]))
    print('Recall (favor): {:.4f}'.format(recall[0]))
    print('F1 Score (favor): {:.4f}'.format(f1[0]))

    print('Precision (neutral): {:.4f}'.format(precision[1]))
    print('Recall (neutral): {:.4f}'.format(recall[1]))
    print('F1 Score (neutral): {:.4f}'.format(f1[1]))

    print('Precision (against): {:.4f}'.format(precision[2]))
    print('Recall (against): {:.4f}'.format(recall[2]))
    print('F1 Score (against): {:.4f}'.format(f1[2]))

    print('Average F1 Score: {:.4f}'.format(avg_f1))

    # Calculate the accuracy
    print(correct_predictions, total_predictions)
    accuracy = correct_predictions.double() / total_predictions.double()

    print('Test Accuracy: {:.4f}'.format(accuracy))


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
    device = get_device()

    weights = [5000, 1, 1]
    loss_function = define_loss_function(weights)

    model = train_model(model, train_loader, optimizer, device, loss_function, num_epochs=15)

    evaluate_model(model, test_loader, device)

    evaluate_reddit(tokenizer, model, device)


if __name__ == '__main__':
    main()