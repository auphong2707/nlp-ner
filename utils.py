import os
import json
import random
import numpy as np
import torch
import shutil
import platform
from functools import partial


from transformers import AutoTokenizer
from datasets import Dataset
from huggingface_hub import hf_hub_download
from typing import Tuple

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
def tokenize_and_align_labels(example, tokenizer):
    tokenized_inputs = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
    )
    label_ids = []
    word_ids = tokenized_inputs.word_ids(batch_index=0)
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(31)
        elif word_idx != previous_word_idx:
            label_ids.append(example["ner_tags"][word_idx])
        else:
            label_ids.append(31)
        previous_word_idx = word_idx
    # Set the label for [CLS] to 0
    if len(label_ids) > 0:
        label_ids[0] = 0  # Assign 'O' or a valid label to [CLS]
    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs

def prepare_dataset(tokenizer_name) -> Tuple[Dataset, Dataset, Dataset, AutoTokenizer]:
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
    num_proc = 2 if platform.system() == "Windows" else 4
    # num_proc = 2
    tokenization_fn = partial(tokenize_and_align_labels, tokenizer=TOKENIZER)

    train_dataset = train_dataset.map(tokenization_fn, batched=False, remove_columns=train_dataset.column_names,num_proc=num_proc)
    val_dataset = val_dataset.map(tokenization_fn, batched=False, remove_columns=val_dataset.column_names,num_proc=num_proc)
    test_dataset = test_dataset.map(tokenization_fn, batched=False, remove_columns=test_dataset.column_names,num_proc=num_proc)
    
    return train_dataset, val_dataset, test_dataset, TOKENIZER

if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset, TOKENIZER = prepare_dataset("dbmdz/bert-large-cased-finetuned-conll03-english")
    print("Train example\n", train_dataset[0])
    print("Validation example \n", val_dataset[0])
    print("Test example \n", test_dataset[0]) 