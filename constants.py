"""
This file contains all the constants used in the project.
"""

RESULTS_DIR = "./results/"
LOGGING_STEPS = 86
SEED = 42

# [I - CODE SUMMARIZATION]
RESULTS_CS_DIR = RESULTS_DIR + "code-summarization/"
CS_INPUT_MAX_LENGTH = 256
CS_OUTPUT_MAX_LENGTH = 128
CS_NUM_BEAMS = 5

# [1. CODE BERT HYPERPARAMETERS]
MODEL_BERT_NER = "dbmdz/bert-large-cased-finetuned-conll03-english"
TOKENIZER_BERT = "dbmdz/bert-large-cased-finetuned-conll03-english"
RESULTS_BERT_DIR = RESULTS_CS_DIR + "code-bert/"

LR_BERT = 8e-5
TRAIN_BATCH_SIZE_BERT = 8
EVAL_BATCH_SIZE_BERT = 8
NUM_TRAIN_EPOCHS_BERT = 50
WEIGHT_DECAY_BERT = 0.01
