"""
This file contains all the constants used in the project.
"""

RESULTS_DIR = "./results/"
LOGGING_STEPS = 86
SEED = 42

# [1. CODE BERT HYPERPARAMETERS]
MODEL_BERT_NER = "google-bert/bert-base-cased"
TOKENIZER_BERT = "google-bert/bert-base-cased"
RESULTS_BERT_DIR = RESULTS_DIR + "bert/"

LR_BERT = 8e-5
TRAIN_BATCH_SIZE_BERT = 8
EVAL_BATCH_SIZE_BERT = 8
NUM_TRAIN_EPOCHS_BERT = 10
WEIGHT_DECAY_BERT = 0.01
