"""
This file contains all the constants used in the project.
"""

RESULTS_DIR = "./results/"
LOGGING_STEPS = 3
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

# Label to ID mapping
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}

# [1. BERT CLASSIFICATION HYPERPARAMETERS]
MODEL_BERT_NER = "google-bert/bert-base-cased"
TOKENIZER_BERT = "google-bert/bert-base-cased"

RESULTS_BERT_DIR = RESULTS_DIR + "bert-cls/"
EXPERIMENT_NAME = "bert-cls-experiment-1"
EXPERIMENT_RESULTS_DIR = RESULTS_BERT_DIR + EXPERIMENT_NAME

LR_BERT = 8e-5
TRAIN_BATCH_SIZE_BERT = 16
EVAL_BATCH_SIZE_BERT = 16
NUM_TRAIN_EPOCHS_BERT = 1
WEIGHT_DECAY_BERT = 0.01
EVAL_STEPS_BERT = 1641
SAVE_STEPS_BERT = 1641

# [2. T5 CLASSIFICATION HYPERPARAMETERS]
MODEL_T5 = "google/flan-t5-base"
TOKENIZER_T5 = "google/flan-t5-base"

RESULTS_T5_DIR = "./results/t5-cls/"
EXPERIMENT_NAME_T5 = "t5-cls-experiment"
EXPERIMENT_RESULTS_DIR_T5 = RESULTS_T5_DIR + EXPERIMENT_NAME_T5

LR_T5 = 3e-5
TRAIN_BATCH_SIZE_T5 = 8
EVAL_BATCH_SIZE_T5 = 8
NUM_TRAIN_EPOCHS_T5 = 3
WEIGHT_DECAY_T5 = 0.01
EVAL_STEPS_T5 = 3
SAVE_STEPS_T5 = 3

