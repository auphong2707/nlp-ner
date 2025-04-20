from transformers import AutoTokenizer
from utils import constants
from models.model_t5_crf import T5CRF
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("test/", local_files_only=True)
model = T5CRF.from_pretrained("test/")
model.eval()

sentence = "Duc Minh Vu is very handsome."
inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Forward pass
with torch.no_grad():
    output = model(**inputs)
    predictions = output["predictions"][0]  # batch_size = 1

# Convert token IDs back to tokens
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
predicted_labels = [constants.ID2LABEL.get(p, "O") for p in predictions]

print("Tokens:", tokens)
print("Predicted Labels:", predicted_labels)

# Clean up subwords and special tokens
final_labels = []
for i, token in enumerate(tokens):
    if token in tokenizer.all_special_tokens:
        continue
    if not token.startswith("▁") and final_labels:  # For sentencepiece tokenizer (like T5)
        continue
    final_labels.append(predicted_labels[i])

print("Final Labels:", final_labels)
