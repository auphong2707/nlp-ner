from transformers import BertTokenizer, AutoModelForTokenClassification
from utils import constants

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("test/", local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained("test/")

sentence = "Manchester United will face Liverpool in the Premier League on Sunday."
inputs = tokenizer(sentence, return_tensors="pt")
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)  # Choose the highest logit as the prediction
predicted_labels = [constants.ID2LABEL.get(pred.item(), "UNKNOWN") for pred in predictions[0]]

# print("Number of tokens:", len(inputs["input_ids"][0]))
# print("Number of prediction tokens:", len(predictions[0]))
# print(predictions)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print("Tokens:", tokens)
print("Predicted Labels:", predicted_labels)

final_labels = []

for i, token in enumerate(tokens):
    # Skip special tokens like [CLS] and [SEP]
    if token in ['[CLS]', '[SEP]']:
        continue
    
    # Handle subwords (tokens with '##' are part of the same word)
    if '##' not in token:
        # For the first token of a word, add the corresponding label
        final_labels.append(predicted_labels[i])

# Output the final labels corresponding to each word
print("Final Labels:", final_labels)