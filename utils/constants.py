"""
This file contains all the constants used in the project.
"""

import torch


RESULTS_DIR = "./results/"
SEED = 42
GAMMA = 1.0
NUM_LABELS = 31
LOSS_SCALE = 100.0

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
    -100: "PAD"
}

# Label to ID mapping
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}

# NER class weights
NER_CLASS_WEIGHTS = torch.tensor([
    1.8400e-04, 7.5585e-03, 7.1658e-03, 1.6998e-02, 1.2870e-02, 7.2909e-03,
    1.6852e-02, 3.7005e-02, 8.4086e-02, 3.5347e+00, 1.4683e+01, 2.0306e-01,
    3.7648e-01, 5.1329e-02, 7.5774e-02, 1.8030e-01, 1.0705e-01, 5.2223e-02,
    1.4170e-01, 1.3473e+00, 1.4424e+00, 7.6780e-02, 4.6879e-02, 8.7157e-01,
    4.8942e+00, 6.0213e-02, 1.8098e-01, 1.8173e-01, 2.3536e-01, 1.1097e+00,
    9.5756e-01
], dtype=torch.float32)


# [1. BERT CLASSIFICATION CROSS-ENTROPY LOSS HYPERPARAMETERS]
MODEL_BERT_CLS_CE = "google-bert/bert-base-cased"
TOKENIZER_BERT_CLS_CE = "google-bert/bert-base-cased"

RESULTS_DIR_BERT_CLS_CE = RESULTS_DIR + "bert-cls-ce/"
EXPERIMENT_NAME_BERT_CLS_CE = "bert-cls-ce-"
EXPERIMENT_RESULTS_DIR_BERT_CLS_CE = RESULTS_DIR_BERT_CLS_CE + EXPERIMENT_NAME_BERT_CLS_CE

LR_BERT_CLS_CE = 1e-4
TRAIN_BATCH_SIZE_BERT_CLS_CE = 64
EVAL_BATCH_SIZE_BERT_CLS_CE = 64
NUM_TRAIN_EPOCHS_BERT_CLS_CE = 5
WEIGHT_DECAY_BERT_CLS_CE = 0.01
EVAL_STEPS_BERT_CLS_CE = 100
SAVE_STEPS_BERT_CLS_CE = 100
GRADIENT_ACCUMULATION_STEPS_BERT_CLS_CE = 1

LOGGING_STEPS_BERT_CLS_CE = 100
# --------------------------------------------------------------------------------------------


# [2. T5 CLASSIFICATION CROSS-ENTROPY LOSS HYPERPARAMETERS]
MODEL_T5_CLS_CE = "t5-base"  
TOKENIZER_T5_CLS_CE = "t5-base"  

RESULTS_DIR_T5_CLS_CE = RESULTS_DIR + "t5-cls-ce/"
EXPERIMENT_NAME_T5_CLS_CE = "t5-cls-ce-"
EXPERIMENT_RESULTS_DIR_T5_CLS_CE = RESULTS_DIR_T5_CLS_CE + EXPERIMENT_NAME_T5_CLS_CE

LR_T5_CLS_CE = 2e-5               
TRAIN_BATCH_SIZE_T5_CLS_CE = 8    
EVAL_BATCH_SIZE_T5_CLS_CE = 8     
NUM_TRAIN_EPOCHS_T5_CLS_CE = 3    
WEIGHT_DECAY_T5_CLS_CE = 0.001    
EVAL_STEPS_T5_CLS_CE = 500       
SAVE_STEPS_T5_CLS_CE = 500        
GRADIENT_ACCUMULATION_STEPS_T5_CLS_CE = 1

LOGGING_STEPS_T5_CLS_CE = 500
# --------------------------------------------------------------------------------------------


# [3. ROBERTA CLASSIFICATION CROSS-ENTROPY LOSS HYPERPARAMETERS]
MODEL_ROBERTA_CLS_CE = "roberta-base"
TOKENIZER_ROBERTA_CLS_CE = "roberta-base"

RESULTS_DIR_ROBERTA_CLS_CE = RESULTS_DIR + "roberta-cls-ce/"
EXPERIMENT_NAME_ROBERTA_CLS_CE = "roberta-cls-ce-"
EXPERIMENT_RESULTS_DIR_ROBERTA_CLS_CE = RESULTS_DIR_ROBERTA_CLS_CE + EXPERIMENT_NAME_ROBERTA_CLS_CE

LR_ROBERTA_CLS_CE = 1e-5
TRAIN_BATCH_SIZE_ROBERTA_CLS_CE = 32
EVAL_BATCH_SIZE_ROBERTA_CLS_CE = 32
NUM_TRAIN_EPOCHS_ROBERTA_CLS_CE = 20
WEIGHT_DECAY_ROBERTA_CLS_CE = 0.01
EVAL_STEPS_ROBERTA_CLS_CE = 373
SAVE_STEPS_ROBERTA_CLS_CE = 373
GRADIENT_ACCUMULATION_STEPS_ROBERTA_CLS_CE = 1

LOGGING_STEPS_ROBERTA_CLS_CE = 373
# --------------------------------------------------------------------------------------------


# [4. BERT CLASSIFICATION FOCAL LOSS HYPERPARAMETERS]
MODEL_BERT_CLS_FOCAL = "google-bert/bert-base-cased"
TOKENIZER_BERT_CLS_FOCAL = "google-bert/bert-base-cased"

RESULTS_DIR_BERT_CLS_FOCAL = RESULTS_DIR + "bert-cls-focal/"
EXPERIMENT_NAME_BERT_CLS_FOCAL = "bert-cls-focal-"
EXPERIMENT_RESULTS_DIR_BERT_CLS_FOCAL = RESULTS_DIR_BERT_CLS_FOCAL + EXPERIMENT_NAME_BERT_CLS_FOCAL

LR_BERT_CLS_FOCAL = 1e-4
TRAIN_BATCH_SIZE_BERT_CLS_FOCAL = 64
EVAL_BATCH_SIZE_BERT_CLS_FOCAL = 64
NUM_TRAIN_EPOCHS_BERT_CLS_FOCAL = 5
WEIGHT_DECAY_BERT_CLS_FOCAL = 0.01
EVAL_STEPS_BERT_CLS_FOCAL = 100
SAVE_STEPS_BERT_CLS_FOCAL = 100
GRADIENT_ACCUMULATION_STEPS_BERT_CLS_FOCAL = 2

LOGGING_STEPS_BERT_CLS_FOCAL = 100
# --------------------------------------------------------------------------------------------


# [5. T5 CLASSIFICATION FOCAL LOSS HYPERPARAMETERS]
MODEL_T5_CLS_FOCAL = "t5-base"  
TOKENIZER_T5_CLS_FOCAL = "t5-base"  

RESULTS_DIR_T5_CLS_FOCAL = RESULTS_DIR + "t5-cls-focal/"
EXPERIMENT_NAME_T5_CLS_FOCAL = "t5-cls-focal-"
EXPERIMENT_RESULTS_DIR_T5_CLS_FOCAL = RESULTS_DIR_T5_CLS_FOCAL + EXPERIMENT_NAME_T5_CLS_FOCAL

LR_T5_CLS_FOCAL = 2e-5               
TRAIN_BATCH_SIZE_T5_CLS_FOCAL = 8    
EVAL_BATCH_SIZE_T5_CLS_FOCAL = 8     
NUM_TRAIN_EPOCHS_T5_CLS_FOCAL = 3    
WEIGHT_DECAY_T5_CLS_FOCAL = 0.001    
EVAL_STEPS_T5_CLS_FOCAL = 500       
SAVE_STEPS_T5_CLS_FOCAL = 500        
GRADIENT_ACCUMULATION_STEPS_T5_CLS_FOCAL = 1

LOGGING_STEPS_T5_CLS_FOCAL = 500
# --------------------------------------------------------------------------------------------


# [6. ROBERTA CLASSIFICATION FOCAL LOSS HYPERPARAMETERS]
MODEL_ROBERTA_CLS_FOCAL = "roberta-base"
TOKENIZER_ROBERTA_CLS_FOCAL = "roberta-base"

RESULTS_DIR_ROBERTA_CLS_FOCAL = RESULTS_DIR + "roberta-cls-focal/"
EXPERIMENT_NAME_ROBERTA_CLS_FOCAL = "roberta-cls-focal-"
EXPERIMENT_RESULTS_DIR_ROBERTA_CLS_FOCAL = RESULTS_DIR_ROBERTA_CLS_FOCAL + EXPERIMENT_NAME_ROBERTA_CLS_FOCAL

LR_ROBERTA_CLS_FOCAL = 1e-5
TRAIN_BATCH_SIZE_ROBERTA_CLS_FOCAL = 32
EVAL_BATCH_SIZE_ROBERTA_CLS_FOCAL = 32
NUM_TRAIN_EPOCHS_ROBERTA_CLS_FOCAL = 20
WEIGHT_DECAY_ROBERTA_CLS_FOCAL = 0.01
EVAL_STEPS_ROBERTA_CLS_FOCAL = 373
SAVE_STEPS_ROBERTA_CLS_FOCAL = 373
GRADIENT_ACCUMULATION_STEPS_ROBERTA_CLS_FOCAL = 1

LOGGING_STEPS_ROBERTA_CLS_FOCAL = 373
# --------------------------------------------------------------------------------------------


# [7. BERT CRF HYPERPARAMETERS]
MODEL_BERT_CRF = "google-bert/bert-base-cased"
TOKENIZER_BERT_CRF = "google-bert/bert-base-cased"

RESULTS_DIR_BERT_CRF = RESULTS_DIR + "bert-crf/"
EXPERIMENT_NAME_BERT_CRF = "bert+crf-experiment-1"
EXPERIMENT_RESULTS_DIR_BERT_CRF = RESULTS_DIR_BERT_CRF + EXPERIMENT_NAME_BERT_CRF

LR_BERT_CRF = 1e-4
TRAIN_BATCH_SIZE_BERT_CRF = 32
EVAL_BATCH_SIZE_BERT_CRF = 32
NUM_TRAIN_EPOCHS_BERT_CRF = 20
WEIGHT_DECAY_BERT_CRF = 0.01
EVAL_STEPS_BERT_CRF = 373
SAVE_STEPS_BERT_CRF = 373
GRADIENT_ACCUMULATION_STEPS_BERT_CRF = 1

LOGGING_STEPS_BERT_CRF = 373
# --------------------------------------------------------------------------------------------


# [8. ROBERTA CRF HYPERPARAMETERS]
MODEL_ROBERTA_CRF = "roberta-base"  # Hoặc model bạn muốn dùng
TOKENIZER_ROBERTA_CRF = "roberta-base"

RESULTS_ROBERTA_CRF_DIR = RESULTS_DIR + "roberta-crf/"
EXPERIMENT_NAME_ROBERTA_CRF = "roberta+crf-experiment-1"
EXPERIMENT_RESULTS_DIR_ROBERTA_CRF = RESULTS_ROBERTA_CRF_DIR + EXPERIMENT_NAME_ROBERTA_CRF

LR_ROBERTA_CRF = 1e-5
TRAIN_BATCH_SIZE_ROBERTA_CRF = 32
EVAL_BATCH_SIZE_ROBERTA_CRF = 32
NUM_TRAIN_EPOCHS_ROBERTA_CRF = 20
WEIGHT_DECAY_ROBERTA_CRF = 0.01
EVAL_STEPS_ROBERTA_CRF = 373
SAVE_STEPS_ROBERTA_CRF = 373
GRADIENT_ACCUMULATION_STEPS_ROBERTA_CRF = 1

LOGGING_STEPS_ROBERTA_CRF = 373
# --------------------------------------------------------------------------------------------


# [9. T5 CRF HYPERPARAMETERS]
MODEL_T5_CRF = "t5-base"
TOKENIZER_T5_CRF = "t5-base"

RESULTS_DIR_T5_CRF = RESULTS_DIR + "t5-crf/"
EXPERIMENT_NAME_T5_CRF = "T5+CRF-experiment-third"
EXPERIMENT_RESULTS_DIR_T5_CRF = RESULTS_DIR_T5_CRF + EXPERIMENT_NAME_T5_CRF

LR_T5_CRF = 7e-5
TRAIN_BATCH_SIZE_T5_CRF = 128
EVAL_BATCH_SIZE_T5_CRF = 128
NUM_TRAIN_EPOCHS_T5_CRF = 35
WEIGHT_DECAY_T5_CRF = 0.001
EVAL_STEPS_T5_CRF = 100      
SAVE_STEPS_T5_CRF = 100
GRADIENT_ACCUMULATION_STEPS_T5_CRF = 1

LOGGING_STEPS_T5_CRF = 10    

