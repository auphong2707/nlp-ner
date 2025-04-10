import os
import json
import random
import numpy as np
import torch
import shutil

from transformers import AutoTokenizer
from datasets import Dataset
from huggingface_hub import hf_hub_download
from typing import Tuple
from constants import ID2LABEL

# Set seed for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU training
    torch.backends.cudnn.deterministic = True  # Ensures reproducibility in CNNs
    torch.backends.cudnn.benchmark = False  # Disables auto-tuning for reproducibility

TOKENIZER = None

def load_jsonl(file_path) -> list:
    """
    Load a JSONL (JSON Lines) file and return its contents as a list of dictionaries.

    Args:
        file_path (str): The path to the JSONL file to be loaded.

    Returns:
        list: A list of dictionaries, each representing a JSON object from a line in the file.
    """
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

# Preprocessing function: Tokenize and align labels
def tokenize_and_align_labels(example):
    # Tokenize while telling the tokenizer that the input is already split into words.
    tokenized_inputs = TOKENIZER(example["tokens"], truncation=True, padding="max_length", is_split_into_words=True)
    labels = []
    word_ids = tokenized_inputs.word_ids()  # Map tokens back to word indices
    previous_word_idx = None
    for word_idx in word_ids:
        # Special tokens have a word_id of None, so set the label to -100 to ignore them during loss computation.
        if word_idx is None:
            labels.append(-100)
        # For the first token of a given word, assign the label.
        elif word_idx != previous_word_idx:
            labels.append(example["ner_tags"][word_idx])
        # For subsequent tokens in a word, assign -100 so that we only predict once per word.
        else:
            labels.append(-100)
        previous_word_idx = word_idx
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def prepare_dataset(tokenizer_name, add_prefix_space=False) -> Tuple[Dataset, Dataset, Dataset, AutoTokenizer]:
    """
    Prepares the dataset for training and validation.

    This function performs the following steps:
    1. Downloads the dataset from the Hugging Face repository.
    2. Loads the training and validation data from JSONL files.
    3. Converts the loaded data into Dataset objects.
    4. Tokenizes the data using a specified tokenizer.

    Returns:
        tuple: A tuple containing the tokenized training dataset, validation dataset, and the tokenizer.
    """
    # Load the tokenizer
    global TOKENIZER
    if add_prefix_space:
        TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=add_prefix_space)
    else:
        TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load data and split it into 3 sets: train, validation, and test
    if not os.path.exists("data"):
        os.mkdir("data")

    if not os.path.exists("data/train_en.jsonl") or not os.path.exists("data/val_en.jsonl") or not os.path.exists("data/test_en.jsonl"):
        file_path = hf_hub_download(repo_id="Babelscape/multinerd", filename="train/train_en.jsonl", repo_type="dataset", local_dir="data/")
        file_path = hf_hub_download(repo_id="Babelscape/multinerd", filename="val/val_en.jsonl", repo_type="dataset", local_dir="data/")
        file_path = hf_hub_download(repo_id="Babelscape/multinerd", filename="test/test_en.jsonl", repo_type="dataset", local_dir="data/")

        shutil.rmtree(os.path.join("data", ".cache"))

        for root, dirs, files in os.walk("data"):
            for file in files:
                shutil.move(os.path.join(root, file), "data")

        for root, dirs, files in os.walk("data"):
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))

    # Load data
    train_data = load_jsonl("data/train_en.jsonl")
    val_data = load_jsonl("data/val_en.jsonl")
    test_data = load_jsonl("data/test_en.jsonl")

    # Randomly sample 20% of data
    # train_data = random.sample(train_data, int(0.2 * len(train_data)))
    # val_data = random.sample(val_data, int(0.2 * len(val_data)))
    # test_data = random.sample(test_data, int(0.2 * len(test_data)))

    # Convert data into a Dataset object
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    # Tokenize the data
    train_dataset = train_dataset.map(tokenize_and_align_labels, batched=False, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(tokenize_and_align_labels, batched=False, remove_columns=val_dataset.column_names)
    test_dataset = test_dataset.map(tokenize_and_align_labels, batched=False, remove_columns=test_dataset.column_names)
    
    return train_dataset, val_dataset, test_dataset, TOKENIZER

# Tokenization and alignment for T5
def tokenize_t5(example):
    tokenized_inputs = TOKENIZER(
        example["tokens"],  # No need to concatenate strings, keep as a list of tokens
        truncation=True,
        padding="max_length",
        max_length=128,
        is_split_into_words=True  # Ensure word-to-token alignment
    )

    # Assign labels to the tokenized sequence
    labels = [-100] * len(tokenized_inputs["input_ids"])  # Default to -100 to ignore special tokens
    word_ids = tokenized_inputs.word_ids()  # Map token to the original word

    for i, word_id in enumerate(word_ids):
        if word_id is not None:  # Ignore special tokens
            labels[i] = example["ner_tags"][word_id]

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Prepare dataset for T5
def prepare_dataset_t5(tokenizer_name):
    global TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load data and split it into 3 sets: train, validation, and test
    if not os.path.exists("data"):
        os.mkdir("data")

    if not os.path.exists("data/train_en.jsonl") or not os.path.exists("data/val_en.jsonl") or not os.path.exists("data/test_en.jsonl"):
        file_path = hf_hub_download(repo_id="Babelscape/multinerd", filename="train/train_en.jsonl", repo_type="dataset", local_dir="data/")
        file_path = hf_hub_download(repo_id="Babelscape/multinerd", filename="val/val_en.jsonl", repo_type="dataset", local_dir="data/")
        file_path = hf_hub_download(repo_id="Babelscape/multinerd", filename="test/test_en.jsonl", repo_type="dataset", local_dir="data/")

        shutil.rmtree(os.path.join("data", ".cache"))

        for root, dirs, files in os.walk("data"):
            for file in files:
                shutil.move(os.path.join(root, file), "data")

        for root, dirs, files in os.walk("data"):
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))

    train_data = load_jsonl("data/train_en.jsonl")
    val_data = load_jsonl("data/val_en.jsonl")
    test_data = load_jsonl("data/test_en.jsonl")

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)

    train_dataset = train_dataset.map(tokenize_t5, batched=False, remove_columns=["tokens", "ner_tags"])
    val_dataset = val_dataset.map(tokenize_t5, batched=False, remove_columns=["tokens", "ner_tags"])
    test_dataset = test_dataset.map(tokenize_t5, batched=False, remove_columns=["tokens", "ner_tags"])

    return train_dataset, val_dataset, test_dataset, TOKENIZER


if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset, TOKENIZER = prepare_dataset_t5("google/flan-t5-base")
    print("Train example\n", train_dataset[0])
    print("Validation example \n", val_dataset[0])
    print("Test example \n", test_dataset[0]) 