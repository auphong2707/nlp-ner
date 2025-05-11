from transformers import (
    BertTokenizer, BertForTokenClassification,
    RobertaTokenizer, RobertaForTokenClassification,
    T5Tokenizer, T5ForConditionalGeneration,
    pipeline
)

_loaded_models = {}

HF_REPO_ID = "auphong2707/nlp-ner"

# Match these EXACTLY to the Hugging Face subfolders
MODEL_CONFIG = {
    "bert_cls_focal_1": {"subfolder": "bert-cls-focal-experiment-1", "type": "bert"},
    "bert_cls_focal_2": {"subfolder": "bert-cls-focal-experiment-2", "type": "bert"},
    "bert_ce_2": {"subfolder": "bert-cls-ce-experiment-2", "type": "bert"},
    "roberta_ce_2": {"subfolder": "roberta-cls-ce-experiment-2", "type": "roberta"},
    "roberta_focal_1": {"subfolder": "roberta-cls-focal-experiment-1", "type": "roberta"},
    "t5_crf_1": {"subfolder": "T5+CRF-experiment-1", "type": "t5"},
    "t5_crf_2": {"subfolder": "T5+CRF-experiment-2", "type": "t5"},
    "t5_crf_first": {"subfolder": "T5+CRF-experiment-first", "type": "t5"},
    "t5_crf_second": {"subfolder": "T5+CRF-experiment-second", "type": "t5"},
    "t5_crf_third": {"subfolder": "T5+CRF-experiment-third", "type": "t5"},
}

def load_model(model_key):
    if model_key in _loaded_models:
        return _loaded_models[model_key]

    model_info = MODEL_CONFIG[model_key]
    model_type = model_info["type"]
    subfolder = model_info["subfolder"]

    if model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(HF_REPO_ID, subfolder=subfolder)
        model = BertForTokenClassification.from_pretrained(HF_REPO_ID, subfolder=subfolder)
        ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

        def predict_fn(text):
            return ner(text)

    elif model_type == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(HF_REPO_ID, subfolder=subfolder)
        model = RobertaForTokenClassification.from_pretrained(HF_REPO_ID, subfolder=subfolder)
        ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

        def predict_fn(text):
            return ner(text)

    elif model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained(HF_REPO_ID, subfolder=subfolder)
        model = T5ForConditionalGeneration.from_pretrained(HF_REPO_ID, subfolder=subfolder)

        def predict_fn(text):
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            outputs = model.generate(**inputs)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return [{"entity": ent} for ent in decoded[0].split()]  # crude formatting

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    _loaded_models[model_key] = predict_fn
    return predict_fn

def get_model_keys():
    return list(MODEL_CONFIG.keys())
