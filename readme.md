# EECS 487 Final Project

## Description

This project is designed to evaluate the feasibility of accurate cross-domain stance detection while utilizing data only from a single domain. There are two datasets included, SemEval-2016 Task 6 data and data from Reddit's /r/unpopularopinion, /r/trueunpopularopinion, and /r/rant. The target for stance detection is the idea that "Climate Change is a Real Concern" and there are three labels: "FAVOR", "AGAINST", "NONE".

It includes two different models for this task: a Naive Bayes classifier and a pre-trained BERT model. The Naive Bayes classifier provides a baseline for performance, while the pre-trained BERT model is fine-tuned to the twitter data and is used to evaluate cross-domain performance. This allows for testing how well a model may generalize to new, unseen data from a different source.

The project is useful for researchers and developers working on stance detection, as well as those interested in cross-domain performance of machine learning models.

## Usage

To use this project, you simply need to run the `bert.py` script. This script will train the BERT model on the SemEval-2016 Task 6 data and evaluate its performance on the test data.

You can run the script from the command line as follows:

```bash
python bert.py
```

This will start the training process which WILL take time since we are using very long training examples, once complete, the script will output the model's performance metrics for both Twitter and Reddit data.