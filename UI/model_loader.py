from transformers import (
    BertTokenizer, BertForTokenClassification,
    RobertaTokenizer, RobertaForTokenClassification,
    T5Tokenizer, T5ForConditionalGeneration
)
from typing import Dict, Callable
import torch
import torch.nn.functional as F

_loaded_models: Dict[str, Callable] = {}

HF_REPO_ID = "auphong2707/nlp-ner"

MODEL_CONFIG = {
    "bert_cls_focal_1": {"subfolder": "bert-cls-focal-experiment-1", "type": "bert"},
    "bert_cls_focal_2": {"subfolder": "bert-cls-focal-experiment-2", "type": "bert"},
    "bert_cls_ce":      {"subfolder": "bert-cls-ce-experiment-2", "type": "bert"},
    "roberta_cls_ce_2": {"subfolder": "roberta-cls-ce-experiment-2", "type": "roberta"},
    "roberta_cls_focal_1": {"subfolder": "roberta-cls-focal-experiment-1", "type": "roberta"},
    "t5_crf":           {"subfolder": "T5+CRF-experiment-first", "type": "t5"},
    "bert_crf":         {"subfolder": "bert+crf-experiment-6", "type": "bert"},
    "t5_cls_ce":         {"subfolder": "t5-cls-ce-experiment-1", "type": "t5"},
}


def load_model(model_key: str):
    if model_key in _loaded_models:
        return _loaded_models[model_key]

    cfg = MODEL_CONFIG[model_key]
    subfolder = cfg["subfolder"]
    model_type = cfg["type"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type in ["bert", "roberta"]:
        tokenizer_cls = BertTokenizer if model_type == "bert" else RobertaTokenizer
        model_cls = BertForTokenClassification if model_type == "bert" else RobertaForTokenClassification

        tokenizer = tokenizer_cls.from_pretrained(HF_REPO_ID, subfolder=subfolder, use_fast=True)
        model = model_cls.from_pretrained(HF_REPO_ID, subfolder=subfolder).to(device).eval()
        id2label = model.config.id2label

        def predict_fn(text: str):
            encoding = tokenizer(text, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                outputs = model(**encoding)
            logits = outputs.logits
            predictions = torch.argmax(F.softmax(logits, dim=-1), dim=-1)[0].tolist()
            tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
            labels = [id2label[idx] for idx in predictions]

            return {
                "tokens": tokens,
                "labels": labels
            }

        _loaded_models[model_key] = predict_fn
        return predict_fn

    elif model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained(HF_REPO_ID, subfolder=subfolder)  # no use_fast here
        model = T5ForConditionalGeneration.from_pretrained(HF_REPO_ID, subfolder=subfolder).to(device).eval()

        def predict_fn(text: str):
            input_text = f"ner: {text}"
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=128)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            # Assume T5 returns: token1|label1 token2|label2 ...
            tokens_labels = [pair.split('|') for pair in decoded.strip().split()]
            tokens, labels = zip(*tokens_labels) if tokens_labels and len(tokens_labels[0]) == 2 else ([], [])
            return {
                "tokens": list(tokens),
                "labels": list(labels)
            }

        _loaded_models[model_key] = predict_fn
        return predict_fn

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_keys():
    return list(MODEL_CONFIG.keys())
