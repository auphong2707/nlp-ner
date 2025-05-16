import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, T5ForTokenClassification, T5Config
from typing import Tuple, List, Union

class ModelLoader:
    def __init__(self, huggingface_repo: str = "auphong2707/nlp-ner"):
        """
        Initialize the ModelLoader with a Hugging Face repository.
        
        Args:
            huggingface_repo (str): The Hugging Face repository name (e.g., 'auphong2707/nlp-ner').
        """
        self.huggingface_repo = huggingface_repo
        self.available_models = [
            "bert-cls-ce-experiment-2",
            "TS+CRF-first",
            "bert-cls-focal-experiment-2",
            "roberta-cls-focal-experiment-1",
            "roberta-cls-ce-experiment-2",
            "bert+crf-experiment-5",
            "ts-cls-ce-experiment-1",
            "ts-cls-focal-experiment-(5e-4)",
            "roberta+crf-experiment-3"
        ]
        # Mapping of model names to their expected architectures
        self.model_type_map = {
            "bert-cls-ce-experiment-2": "bert",
            "TS+CRF-first": "t5",
            "bert-cls-focal-experiment-2": "bert",
            "roberta-cls-focal-experiment-1": "roberta",
            "roberta-cls-ce-experiment-2": "roberta",
            "bert+crf-experiment-5": "bert",
            "ts-cls-ce-experiment-1": "t5",
            "ts-cls-focal-experiment-(5e-4)": "t5",
            "roberta+crf-experiment-3": "roberta"
        }
        self.model = None
        self.tokenizer = None
        self.loaded_model_name = None

    def load_model(self, model_name: str) -> Tuple[Union[AutoModelForTokenClassification, T5ForTokenClassification], AutoTokenizer]:
        """
        Load a specific model and tokenizer from the Hugging Face repository.
        
        Args:
            model_name (str): The name of the model to load (e.g., 'bert-cls-ce-experiment-2').
            
        Returns:
            Tuple containing the loaded model and tokenizer.
            
        Raises:
            ValueError: If the specified model name is not in the available models list.
            Exception: If there's an error loading the model or tokenizer.
        """
        if model_name not in self.available_models:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {self.available_models}"
            )

        # Only load the model if it hasn't been loaded or a different model is requested
        if self.loaded_model_name != model_name:
            try:
                print(f"Loading model from {self.huggingface_repo}, subfolder: {model_name}...")
                model_type = self.model_type_map.get(model_name, "unknown")

                if model_type in ["bert", "roberta"]:
                    self.model = AutoModelForTokenClassification.from_pretrained(
                        self.huggingface_repo,
                        subfolder=model_name
                    )
                elif model_type == "t5":
                    self.model = T5ForTokenClassification.from_pretrained(
                        self.huggingface_repo,
                        subfolder=model_name
                    )
                else:
                    raise ValueError(f"Unsupported model type for '{model_name}': {model_type}")

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.huggingface_repo,
                    subfolder=model_name
                )
                self.loaded_model_name = model_name
                print(f"Successfully loaded model: {model_name}")
            except Exception as e:
                raise Exception(f"Error loading model '{model_name}': {str(e)}")
        return self.model, self.tokenizer

    def get_available_models(self) -> List[str]:
        """
        Get the list of available models.
        
        Returns:
            List of available model names.
        """
        return self.available_models

if __name__ == "__main__":
    # Example usage
    loader = ModelLoader()
    selected_model = "ts-cls-ce-experiment-1"
    try:
        model, tokenizer = loader.load_model(selected_model)
        print(f"Loaded model: {selected_model}")
        print(f"Model architecture: {model}")
        print(f"Tokenizer: {tokenizer}")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")