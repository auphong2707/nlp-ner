import os
import sys
from typing import Tuple, List, Union

import torch
from torchcrf import CRF
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    T5ForTokenClassification,
    T5Config,
    RobertaPreTrainedModel,
)

# Add the parent directory to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model_t5_crf import T5CRF
from models.model_bert_crf import BertCRF


class RobertaCRF(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = AutoModelForTokenClassification.from_config(config)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        emissions = outputs.logits
        mask = attention_mask.bool() if attention_mask is not None else None
        if labels is not None:
            labels = labels.clone()
            labels[labels == -100] = 0
            loss = -self.crf(emissions, labels, mask=mask, reduction='token_mean')
            return {"loss": loss, "logits": emissions}
        else:
            predictions = self.crf.decode(emissions, mask=mask)
            return {"logits": emissions, "predictions": predictions}


class BaseModelLoader:
    def __init__(self, huggingface_repo: str, model_name: str):
        self.huggingface_repo = huggingface_repo
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load(self):
        raise NotImplementedError("Subclasses must implement the load method.")


class BertModelLoader(BaseModelLoader):
    def load(self):
        if self.model_name == "bert+crf-experiment-5":
            self.model = BertCRF.from_pretrained(self.huggingface_repo, subfolder=self.model_name)
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(self.huggingface_repo, subfolder=self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.huggingface_repo, subfolder=self.model_name)
        return self.model, self.tokenizer


class RobertaModelLoader(BaseModelLoader):
    def load(self):
        if self.model_name == "roberta+crf-experiment-3":
            self.model = RobertaCRF.from_pretrained(self.huggingface_repo, subfolder=self.model_name)
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(self.huggingface_repo, subfolder=self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.huggingface_repo, subfolder=self.model_name)
        return self.model, self.tokenizer


class T5ModelLoader(BaseModelLoader):
    def load(self):
        config = T5Config.from_pretrained(self.huggingface_repo, subfolder=self.model_name)
        config.num_labels = 31
        self.model = T5ForTokenClassification.from_pretrained(
            self.huggingface_repo, subfolder=self.model_name, config=config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.huggingface_repo, subfolder=self.model_name, use_fast=True)
        return self.model, self.tokenizer


class T5FocalModelLoader(BaseModelLoader):
    def load(self):
        repo = "auphong2707/nlp-ner-t5-focal"
        config = T5Config.from_pretrained(repo)
        config.num_labels = 31
        self.model = T5ForTokenClassification.from_pretrained(repo, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(repo, use_fast=True)
        return self.model, self.tokenizer


class T5CRFModelLoader(BaseModelLoader):
    def load(self):
        self.model = T5CRF.from_pretrained(self.huggingface_repo, subfolder=self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.huggingface_repo, subfolder=self.model_name)
        return self.model, self.tokenizer


class ModelLoaderFactory:
    def __init__(self, huggingface_repo: str = "auphong2707/nlp-ner"):
        self.huggingface_repo = huggingface_repo
        self.available_models = [
            "bert-cls-ce-experiment-2",
            "T5+CRF-experiment-first",
            "bert-cls-focal-experiment-2",
            "roberta-cls-focal-experiment-1",
            "roberta-cls-ce-experiment-2",
            "bert+crf-experiment-5",
            "t5-cls-ce-experiment-1",
            "T5-cls-focal-experiment-(5e-4)",
            "roberta+crf-experiment-3"
        ]

        self.model_display_names = {
            "T5+CRF-experiment-first": "T5 + CRF",
            "bert+crf-experiment-5": "BERT + CRF",
            "roberta+crf-experiment-3": "RoBERTa + CRF",
            "t5-cls-ce-experiment-1": "T5-CLS (CE)",
            "roberta-cls-focal-experiment-1": "RoBERTa-CLS (Focal)",
            "roberta-cls-ce-experiment-2": "RoBERTa-CLS (CE)",
            "bert-cls-focal-experiment-2": "BERT-CLS (Focal)",
            "bert-cls-ce-experiment-2": "BERT-CLS (CE)",
            "T5-cls-focal-experiment-(5e-4)": "T5-CLS (Focal)"
        }

        self.model_type_map = {
            "bert": ["bert-cls-ce-experiment-2", "bert-cls-focal-experiment-2", "bert+crf-experiment-5"],
            "roberta": ["roberta-cls-focal-experiment-1", "roberta-cls-ce-experiment-2", "roberta+crf-experiment-3"],
            "t5": ["t5-cls-ce-experiment-1"],
            "t5_focal": ["T5-cls-focal-experiment-(5e-4)"],
            "t5_crf": ["T5+CRF-experiment-first"]
        }

        self.loader_map = {
            "bert": BertModelLoader,
            "roberta": RobertaModelLoader,
            "t5": T5ModelLoader,
            "t5_focal": T5FocalModelLoader,
            "t5_crf": T5CRFModelLoader
        }

    def get_loader(self, internal_name: str) -> BaseModelLoader:
        for model_type, model_list in self.model_type_map.items():
            if internal_name in model_list:
                loader_cls = self.loader_map[model_type]
                return loader_cls(self.huggingface_repo, internal_name)
        raise ValueError(f"No loader found for model: {internal_name}")

    def get_display_models(self) -> List[dict]:
        return [
            {"internal_name": name, "display_name": self.model_display_names.get(name, name)}
            for name in self.available_models
        ]

    def get_available_models(self) -> List[str]:
        return self.available_models


if __name__ == "__main__":
    factory = ModelLoaderFactory()
    print("Testing model loading...\n")
    for model_info in factory.get_display_models():
        try:
            loader = factory.get_loader(model_info["internal_name"])
            loader.load()
            print(f"✅ Loaded: {model_info['display_name']}")
        except Exception as e:
            print(f"❌ Failed: {model_info['display_name']} — {str(e)}")
