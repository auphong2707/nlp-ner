import os
import json
import random
import numpy as np
from sklearn.utils import compute_class_weight
import torch
import shutil

from transformers import AutoTokenizer
from datasets import Dataset
from huggingface_hub import hf_hub_download
from typing import Tuple

from utils.constants import TOKENIZER_BERT_CRF

def set_seed(seed: int = 42):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    This function sets the seed for Python's `random` module, NumPy, and PyTorch.
    It also configures PyTorch to ensure deterministic behavior in computations,
    which is particularly useful for debugging and reproducibility in machine
    learning experiments.

    Args:
        seed (int, optional): The seed value to use for random number generation.
                              Defaults to 42.

    Notes:
        - Setting `torch.backends.cudnn.deterministic` to `True` ensures that
          convolution operations are deterministic, but may reduce performance.
        - Setting `torch.backends.cudnn.benchmark` to `False` disables the
          auto-tuner that selects the best algorithm for the hardware, which
          also helps with reproducibility.
    """
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


def download_dataset():
    """
    Downloads and organizes the MultiNERD dataset into train, validation, and test sets.

    This function checks if the "data" directory exists and creates it if necessary. 
    It then downloads the train, validation, and test JSONL files from the Hugging Face 
    Hub if they are not already present in the "data" directory. After downloading, 
    it removes any unnecessary cache files and reorganizes the directory structure 
    by moving all files to the "data" directory and removing any empty subdirectories.

    Dependencies:
        - os
        - shutil
        - hf_hub_download (from huggingface_hub)

    Raises:
        - Any exceptions raised by `hf_hub_download`, `os`, or `shutil` operations.

    Note:
        Ensure that the `hf_hub_download` function from the `huggingface_hub` library 
        is available in your environment before using this function.
    """
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


def tokenize_and_align_labels(example):
    """
    Tokenizes input text and aligns the corresponding labels for Named Entity Recognition (NER) tasks.

    This function tokenizes the input text while preserving the word boundaries and aligns the NER tags 
    with the tokenized output. Special tokens and subword tokens are ignored during loss computation by 
    assigning them a label of -100.

    Args:
        example (dict): A dictionary containing:
            - "tokens" (list of str): The list of words in the input text.
            - "ner_tags" (list of int): The list of NER tag indices corresponding to each word.

    Returns:
        dict: A dictionary containing:
            - Tokenized inputs with keys such as "input_ids", "attention_mask", etc.
            - "labels" (list of int): The aligned NER tags for the tokenized inputs, with -100 for 
              special tokens and subword tokens.
    """
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
    if len(labels)>0 and TOKENIZER.name_or_path == TOKENIZER_BERT_CRF:
        labels[0]=0 #assign 0 or a valid label to [CLS]
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def prepare_dataset(tokenizer_name, add_prefix_space=False) -> Tuple[Dataset, Dataset, Dataset, AutoTokenizer]:
    """
    Prepares the dataset for training, validation, and testing by loading, processing, 
    and tokenizing the data.
    Args:
        tokenizer_name (str): The name or path of the tokenizer to be loaded.
        add_prefix_space (bool, optional): Whether to add a prefix space for tokenizers 
            that require it (e.g., GPT-2). Defaults to False.
    Returns:
        Tuple[Dataset, Dataset, Dataset, AutoTokenizer]: A tuple containing the processed 
        training dataset, validation dataset, testing dataset, and the loaded tokenizer.
    """
    # Load the tokenizer
    global TOKENIZER
    if add_prefix_space:
        TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=add_prefix_space)
    else:
        TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)


    # Download dataset if not already present
    download_dataset()

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
    train_dataset = train_dataset.map(tokenize_and_align_labels, batched=False, remove_columns=train_dataset.column_names, num_proc=os.cpu_count() - 1)
    val_dataset = val_dataset.map(tokenize_and_align_labels, batched=False, remove_columns=val_dataset.column_names, num_proc=os.cpu_count() - 1)
    test_dataset = test_dataset.map(tokenize_and_align_labels, batched=False, remove_columns=test_dataset.column_names, num_proc=os.cpu_count() - 1)
    
    return train_dataset, val_dataset, test_dataset, TOKENIZER


def tokenize_t5(example):
    """
    Tokenizes input data for a T5 model and aligns the tokenized sequence with NER (Named Entity Recognition) tags.

    Args:
        example (dict): A dictionary containing the following keys:
            - "tokens" (list of str): A list of tokens representing the input text.
            - "ner_tags" (list of int): A list of integer NER tags corresponding to each token.

    Returns:
        dict: A dictionary containing the tokenized inputs with the following keys:
            - "input_ids" (list of int): Token IDs of the input sequence.
            - "attention_mask" (list of int): Attention mask indicating which tokens are padding.
            - "labels" (list of int): Aligned NER tags for each token, with -100 for special tokens and padding.
    """
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


def prepare_dataset_t5(tokenizer_name):
    """
    Prepares datasets for training, validation, and testing using the T5 tokenizer.

    This function initializes a global tokenizer, downloads the dataset if not already present,
    loads the data from JSONL files, converts it into a Hugging Face Dataset format, and tokenizes
    the data for use with the T5 model.

    Args:
        tokenizer_name (str): The name or path of the pre-trained tokenizer to load.

    Returns:
        tuple: A tuple containing the tokenized training, validation, and test datasets, 
               along with the initialized tokenizer.

    Notes:
        - The function assumes the presence of JSONL files named "train_en.jsonl", 
          "val_en.jsonl", and "test_en.jsonl" in the "data" directory.
        - The datasets are tokenized using the `tokenize_t5` function, and the columns 
          "tokens" and "ner_tags" are removed during the process.
        - The number of processes used for tokenization is determined by the number of 
          available CPU cores.
    """
    global TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)

    # Download dataset if not already present
    download_dataset()

    # Load data
    train_data = load_jsonl("data/train_en.jsonl")
    val_data = load_jsonl("data/val_en.jsonl")
    test_data = load_jsonl("data/test_en.jsonl")

    # Change the format of the data
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)

    # Tokenize the data
    train_dataset = train_dataset.map(tokenize_t5, batched=False, remove_columns=["tokens", "ner_tags"], num_proc=os.cpu_count() - 1)
    val_dataset = val_dataset.map(tokenize_t5, batched=False, remove_columns=["tokens", "ner_tags"], num_proc=os.cpu_count() - 1)
    test_dataset = test_dataset.map(tokenize_t5, batched=False, remove_columns=["tokens", "ner_tags"], num_proc=os.cpu_count() - 1)

    return train_dataset, val_dataset, test_dataset, TOKENIZER


def get_class_weights():
    """
    Computes and returns normalized class weights for NER (Named Entity Recognition) tags 
    based on the distribution of labels in the training, validation, and test datasets.
    The function performs the following steps:
    1. Loads the training, validation, and test datasets from JSONL files.
    2. Extracts NER tags (labels) from the datasets.
    3. Computes class weights using the 'balanced' strategy to handle class imbalance.
    4. Normalizes the computed weights so that their sum equals the number of unique classes.
    Returns:
        torch.Tensor: A tensor containing the normalized class weights for each unique class.
    """
    
    train_data = load_jsonl("data/train_en.jsonl")
    val_data = load_jsonl("data/val_en.jsonl")
    test_data = load_jsonl("data/test_en.jsonl")

    def extract_labels(data):
        labels = []
        for item in data:
            labels.extend(item["ner_tags"])
        return labels

    labels = extract_labels(train_data) + extract_labels(val_data) + extract_labels(test_data)
    print(len(labels))

    classes = np.unique(labels)

    weights = compute_class_weight('balanced', classes=classes, y=labels)
    weights = torch.tensor(weights, dtype=torch.float32)

    normalized_weights = weights / weights.sum() * len(classes)

    return normalized_weights