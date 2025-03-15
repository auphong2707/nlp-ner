"""
This file contains all the constants used in the project.
"""

RESULTS_DIR = "./results/"
LOGGING_STEPS = 11
SEED = 42

ID2LABEL = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-ANIM",
    8: "I-ANIM",
    9: "B-BIO",
    10: "I-BIO",
    11: "B-CEL",
    12: "I-CEL",
    13: "B-DIS",
    14: "I-DIS",
    15: "B-EVE",
    16: "I-EVE",
    17: "B-FOOD",
    18: "I-FOOD",
    19: "B-INST",
    20: "I-INST",
    21: "B-MEDIA",
    22: "I-MEDIA",
    23: "B-MYTH",
    24: "I-MYTH",
    25: "B-PLANT",
    26: "I-PLANT",
    27: "B-TIME",
    28: "I-TIME",
    29: "B-VEHI",
    30: "I-VEHI",
    31: "PAD"
}


# [1. BERT CLASSIFICATION HYPERPARAMETERS]
MODEL_BERT_NER = "google-bert/bert-base-cased"
TOKENIZER_BERT = "google-bert/bert-base-cased"

RESULTS_BERT_DIR = RESULTS_DIR + "bert-cls/"
EXPERIMENT_NAME = "bert-cls-experiment-2"
EXPERIMENT_RESULTS_DIR = RESULTS_BERT_DIR + EXPERIMENT_NAME

LR_BERT = 1e-5
TRAIN_BATCH_SIZE_BERT = 32
EVAL_BATCH_SIZE_BERT = 32
NUM_TRAIN_EPOCHS_BERT = 20
WEIGHT_DECAY_BERT = 0.01
EVAL_STEPS_BERT = 373
SAVE_STEPS_BERT = 373


# [2. BERT BiLSTM CRF HYPERPARAMETERS]
MODEL_BERT_BC = "google-bert/bert-base-cased"
TOKENIZER_BERT_BC = "google-bert/bert-base-cased"

RESULTS_BERT_BC_DIR = RESULTS_DIR + "bert-bbc/"
EXPERIMENT_NAME_BERT_BC= "bert-bilstm+crf-experiment-0"
EXPERIMENT_RESULTS_BBC_DIR = RESULTS_BERT_BC_DIR + EXPERIMENT_NAME_BERT_BC

LR_BERT_BC = 1e-6
TRAIN_BATCH_SIZE_BERT_BC = 16
EVAL_BATCH_SIZE_BERT_BC = 16
NUM_TRAIN_EPOCHS_BERT_BC = 20
WEIGHT_DECAY_BERT_BC = 0.01
EVAL_STEPS_BERT_BC = 373
SAVE_STEPS_BERT_BC = 373