# EECS 487 Final Project

## Description

This project is designed for stance detection on SemEval-2016 Task 6 data and data from Reddit. It includes two different models for this task: a Naive Bayes classifier and a pre-trained BERT model. The Naive Bayes classifier provides a baseline for performance, while the pre-trained BERT model is fine-tuned to the twitter data and is used to evaluate cro--domain performance. This allows for testing how well a model may generalize to new, unseen data from a different source.

The project is useful for researchers and developers working on stance detection, as well as those interested in cross-domain performance of machine learning models.

## Usage

To use this project, you simply need to run the `bert.py` script. This script will train the BERT model on the SemEval-2016 Task 6 data and evaluate its performance on the test data.

You can run the script from the command line as follows:

```bash
python bert.py
```

This will start the training process which WILL take time since we are using a max_length of 512, once complete, the script will output the model's performance metrics for both Twitter and Reddit data.