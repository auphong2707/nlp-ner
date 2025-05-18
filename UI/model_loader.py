import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, T5ForTokenClassification, T5Config, RobertaPreTrainedModel, RobertaConfig
import sys
import os
from typing import Tuple, List, Union
from torchcrf import CRF

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model_t5_crf import T5CRF  # Absolute import after adjusting sys.path
from models.model_bert_crf import BertCRF  # Import the custom BertCRF class

# Updated RobertaCRF to use AutoModelForTokenClassification.from_config
class RobertaCRF(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        print(f"Initializing RobertaCRF with config: model_type={config.model_type}, num_labels={getattr(config, 'num_labels', 'Not specified')}")
        # Use AutoModelForTokenClassification.from_config to instantiate self.roberta
        self.roberta = AutoModelForTokenClassification.from_config(config)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()
        print("RobertaCRF initialized successfully")

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        emissions = outputs.logits  # Shape: (batch_size, sequence_length, num_labels)

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
    """Base class for model loaders."""
    def __init__(self, huggingface_repo: str, model_name: str):
        self.huggingface_repo = huggingface_repo
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load(self) -> Tuple[Union[AutoModelForTokenClassification, T5ForTokenClassification, T5CRF], AutoTokenizer]:
        """Load the model and tokenizer. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the load method.")

class BertModelLoader(BaseModelLoader):
    """Loader for BERT-based models, including custom BertCRF."""
    def load(self) -> Tuple[Union[AutoModelForTokenClassification, BertCRF], AutoTokenizer]:
        try:
            print(f"Loading BERT model from {self.huggingface_repo}, subfolder: {self.model_name}...")
            # Load the model and tokenizer from the root subfolder
            if self.model_name == "bert+crf-experiment-5":
                # Use custom BertCRF for bert+crf-experiment-5
                self.model = BertCRF.from_pretrained(
                    self.huggingface_repo,
                    subfolder=self.model_name
                )
            else:
                # Use standard BertForTokenClassification for other BERT models
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.huggingface_repo,
                    subfolder=self.model_name
                )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.huggingface_repo,
                subfolder=self.model_name
            )
            print(f"Successfully loaded BERT model: {self.model_name}")
            return self.model, self.tokenizer
        except Exception as e:
            raise Exception(f"Error loading BERT model '{self.model_name}': {str(e)}")

class RobertaModelLoader(BaseModelLoader):
    """Loader for RoBERTa-based models, including custom RobertaCRF."""
    def load(self) -> Tuple[Union[AutoModelForTokenClassification, RobertaCRF], AutoTokenizer]:
        try:
            print(f"Loading RoBERTa model from {self.huggingface_repo}, subfolder: {self.model_name}...")
            if self.model_name == "roberta+crf-experiment-3":
                # Use custom RobertaCRF for roberta+crf-experiment-3
                self.model = RobertaCRF.from_pretrained(
                    self.huggingface_repo,
                    subfolder=self.model_name
                )
            else:
                # Use standard RobertaForTokenClassification for other RoBERTa models
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.huggingface_repo,
                    subfolder=self.model_name
                )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.huggingface_repo,
                subfolder=self.model_name
            )
            print(f"Successfully loaded RoBERTa model: {self.model_name}")
            return self.model, self.tokenizer
        except Exception as e:
            raise Exception(f"Error loading RoBERTa model '{self.model_name}': {str(e)}")

class T5ModelLoader(BaseModelLoader):
    """Loader for T5-based models (excluding T5-cls-focal-experiment-(5e-4))."""
    def load(self) -> Tuple[T5ForTokenClassification, AutoTokenizer]:
        try:
            print(f"Loading T5 model from {self.huggingface_repo}, subfolder: {self.model_name}...")
            
            # Use the root subfolder (e.g., t5-cls-ce-experiment-1)
            subfolder = self.model_name

            # Load the configuration
            config = T5Config.from_pretrained(
                self.huggingface_repo,
                subfolder=subfolder
            )
            # Set num_labels to 31 to match checkpoint
            config.num_labels = 31
            print(f"Loaded configuration for {self.model_name}: architectures={config.architectures}, num_labels={config.num_labels}")

            # Load the model with the configuration
            self.model = T5ForTokenClassification.from_pretrained(
                self.huggingface_repo,
                subfolder=subfolder,
                config=config
            )

            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.huggingface_repo,
                subfolder=subfolder,
                use_fast=True
            )
            
            print(f"Successfully loaded T5 model: {self.model_name}")
            return self.model, self.tokenizer
        except Exception as e:
            print(f"Failed to load T5 model '{self.model_name}': {str(e)}")
            raise Exception(f"Error loading T5 model '{self.model_name}': {str(e)}")

class T5FocalModelLoader(BaseModelLoader):
    """Loader specifically for T5-cls-focal-experiment-(5e-4) from a different repository."""
    def load(self) -> Tuple[T5ForTokenClassification, AutoTokenizer]:
        try:
            repo = "auphong2707/nlp-ner-t5-focal"
            print(f"Loading T5 model from {repo}...")
            
            # Load from the root of the repository (no subfolder)
            subfolder = ""

            # Load the configuration
            config = T5Config.from_pretrained(
                repo,
                subfolder=subfolder
            )
            # Set num_labels to 31 to match checkpoint
            config.num_labels = 31
            print(f"Loaded configuration for {self.model_name}: architectures={config.architectures}, num_labels={config.num_labels}")

            # Load the model with the configuration
            self.model = T5ForTokenClassification.from_pretrained(
                repo,
                subfolder=subfolder,
                config=config
            )

            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                repo,
                subfolder=subfolder,
                use_fast=True
            )
            
            print(f"Successfully loaded T5 model: {self.model_name}")
            return self.model, self.tokenizer
        except Exception as e:
            print(f"Failed to load T5 model '{self.model_name}': {str(e)}")
            raise Exception(f"Error loading T5 model '{self.model_name}': {str(e)}")

class T5CRFModelLoader(BaseModelLoader):
    """Loader for T5+CRF-based models."""
    def load(self) -> Tuple[T5CRF, AutoTokenizer]:
        try:
            print(f"Loading T5+CRF model from {self.huggingface_repo}, subfolder: {self.model_name}...")
            # Load the model and tokenizer from the root subfolder
            self.model = T5CRF.from_pretrained(
                self.huggingface_repo,
                subfolder=self.model_name
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.huggingface_repo,
                subfolder=self.model_name
            )
            print(f"Successfully loaded T5+CRF model: {self.model_name}")
            return self.model, self.tokenizer
        except Exception as e:
            raise Exception(f"Error loading T5+CRF model '{self.model_name}': {str(e)}")

class ModelLoaderFactory:
    """Factory class to manage model loaders and available models."""
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
        
        # Mapping for converting model names to official names
        self.model_name_mapping = {
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
            "bert-cls-ce-experiment-2": "bert",
            "T5+CRF-experiment-first": "t5_crf",
            "bert-cls-focal-experiment-2": "bert",
            "roberta-cls-focal-experiment-1": "roberta",
            "roberta-cls-ce-experiment-2": "roberta",
            "bert+crf-experiment-5": "bert",
            "t5-cls-ce-experiment-1": "t5",
            "T5-cls-focal-experiment-(5e-4)": "t5_focal",
            "roberta+crf-experiment-3": "roberta"
        }
        self.loader_map = {
            "bert": BertModelLoader,
            "roberta": RobertaModelLoader,
            "t5": T5ModelLoader,
            "t5_focal": T5FocalModelLoader,
            "t5_crf": T5CRFModelLoader
        }

    def get_loader(self, model_name: str) -> BaseModelLoader:
        if model_name not in self.available_models:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {self.available_models}"
            )
        
        # Map the downloaded model name to the official name
        official_model_name = self.model_name_mapping.get(model_name, model_name)
        print(f"Official Model Name: {official_model_name}")
        
        model_type = self.model_type_map.get(model_name)
        if model_type not in self.loader_map:
            raise ValueError(f"Unsupported model type for '{model_name}': {model_type}")
        
        loader_class = self.loader_map[model_type]
        return loader_class(self.huggingface_repo, official_model_name)

    def get_available_models(self) -> List[str]:
        return self.available_models


if __name__ == "__main__":
    # Test loading all models and only show working models
    factory = ModelLoaderFactory()
    print("Testing loading of all models...")
    working_models = []
    failed_models = []

    for selected_model in factory.get_available_models():
        try:
            loader = factory.get_loader(selected_model)
            model, tokenizer = loader.load()
            working_models.append(selected_model)
        except Exception as e:
            failed_models.append(selected_model)
            print(f"Failed to load {selected_model}: {str(e)}")

    print("\nWorking models:")
    for model in working_models:
        print(f"- {model}")

    print("\nFailed models:")
    for model in failed_models:
        print(f"- {model}")

    print("\nFinished testing all models.")