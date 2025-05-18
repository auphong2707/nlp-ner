from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory (project_directory/) to sys.path to find models/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Add the current directory (UI/) to sys.path to find model_loader
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from model_loader import ModelLoaderFactory  # Import the ModelLoaderFactory

app = FastAPI()
# Use an explicit path for the templates directory
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Initialize the model loader factory
factory = ModelLoaderFactory()
available_models = factory.get_display_models()


# Cache for loaded models
model_cache = {}

# Define the ID2LABEL mapping
ID2LABEL = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-ANIM",
    8: "I-ANIM",
    9: "B-BIO",
    10: "I-BIO",
    11: "B-CEL",
    12: "I-CEL",
    13: "B-DIS",
    14: "I-DIS",
    15: "B-EVE",
    16: "I-EVE",
    17: "B-FOOD",
    18: "I-FOOD",
    19: "B-INST",
    20: "I-INST",
    21: "B-MEDIA",
    22: "I-MEDIA",
    23: "B-MYTH",
    24: "I-MYTH",
    25: "B-PLANT",
    26: "I-PLANT",
    27: "B-TIME",
    28: "I-TIME",
    29: "B-VEHI",
    30: "I-VEHI",
    -100: "PAD"
}

# Simple test route to confirm the server is responding
@app.get("/test")
async def test():
    return {"message": "Server is running!"}

# Function to process input text through the selected model
def process_text(text: str, model_name: str) -> dict:
    try:
        # Load model from cache or initialize
        if model_name not in model_cache:
            loader = factory.get_loader(model_name)
            model, tokenizer = loader.load()
            model_cache[model_name] = (model, tokenizer)
        else:
            model, tokenizer = model_cache[model_name]

        # Split the input text into words
        input_words = text.split()
        logger.info(f"Model {model_name} - Input words: {input_words}")

        # Tokenize the input text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True,
            return_offsets_mapping=True  # To map tokens back to words
        )

        # Log the tokenized input
        logger.info(f"Model {model_name} - Tokenized input IDs: {inputs['input_ids'].tolist()}")
        logger.info(f"Model {model_name} - Attention mask: {inputs['attention_mask'].tolist()}")
        logger.info(f"Model {model_name} - Offset mapping: {inputs['offset_mapping'].tolist()}")

        # Decode tokens for logging
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu().tolist())
        logger.info(f"Model {model_name} - Decoded tokens: {tokens}")

        # Move inputs to the same device as the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Ensure the model is in evaluation mode
        model.eval()

        # Forward pass through the model
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        # Handle model output
        if "predictions" in outputs:  # For CRF-based models (e.g., BertCRF, RobertaCRF)
            predicted_labels = outputs["predictions"][0]  # CRF decode returns a list of lists
            logger.info(f"Model {model_name} - CRF predictions: {predicted_labels}")
        else:  # For non-CRF models (e.g., T5ForTokenClassification)
            logits = outputs.logits  # Shape: (batch_size, sequence_length, num_labels)
            logger.info(f"Model {model_name} - Logits shape: {logits.shape}")
            predicted_labels = torch.argmax(logits, dim=-1).cpu().numpy()[0]  # Take argmax for predictions
            logger.info(f"Model {model_name} - Predicted label IDs: {predicted_labels.tolist()}")

        # Convert label IDs to label names using ID2LABEL
        predicted_labels_str = [ID2LABEL.get(label_id, "O") for label_id in predicted_labels]
        logger.info(f"Model {model_name} - Predicted labels: {predicted_labels_str}")

        # Map tokens back to words
        offset_mapping = inputs["offset_mapping"][0].cpu().numpy()  # Shape: (sequence_length, 2)
        word_labels = []

        # Track the current word index
        word_idx = 0
        current_word_start = 0
        current_word_end = len(text) if not input_words else text.index(input_words[0]) + len(input_words[0])
        current_label = None

        for i, (token, label, (start, end)) in enumerate(zip(tokens, predicted_labels_str, offset_mapping)):
            # Skip special tokens
            if token in tokenizer.all_special_tokens:
                continue

            # Check if this token corresponds to the current word
            word = input_words[word_idx] if word_idx < len(input_words) else None
            if word is None:
                break

            # Adjust the word boundaries
            if start >= current_word_end:
                # We've moved past the current word, assign the label and move to the next word
                if current_label:
                    word_labels.append(current_label)
                else:
                    word_labels.append(label)  # Fallback to the current token's label
                word_idx += 1
                if word_idx < len(input_words):
                    current_word_start = current_word_end + 1 if current_word_end < len(text) else current_word_end
                    next_space = text.find(' ', current_word_start)
                    current_word_end = next_space if next_space != -1 else len(text)
                    current_label = None

            # Aggregate labels for the current word (use the first non-O label if available)
            if label != 'O':
                current_label = label
            elif current_label is None:
                current_label = label

        # Append the label for the last word
        if word_idx < len(input_words) and current_label:
            word_labels.append(current_label)

        logger.info(f"Model {model_name} - Word labels after mapping: {word_labels}")

        # Check for mismatch between input words and word labels
        if len(input_words) != len(word_labels):
            logger.warning(f"Mismatch between input words ({len(input_words)}) and word labels ({len(word_labels)}) for model {model_name}.")
            return {"error": f"Model {model_name} failed: Mismatch between input words ({len(input_words)}) and labels ({len(word_labels)}). Please check the model output."}

        # Pair words with their labels
        token_label_pairs = []
        for word, label in zip(input_words, word_labels):
            # Extract the entity type (e.g., 'PER' from 'B-PER' or 'I-PER')
            entity_type = label.split('-')[1] if '-' in label and label != 'O' else label
            token_label_pairs.append({
                "token": word,
                "label": label,
                "type": entity_type
            })

        # Free up GPU memory
        del model
        del tokenizer
        torch.cuda.empty_cache()

        logger.info(f"Model {model_name} - Processed token-label pairs: {token_label_pairs}")
        return {"token_label_pairs": token_label_pairs}
    except Exception as e:
        logger.error(f"Error in process_text: {str(e)}")
        return {"error": f"Failed to process text: {str(e)}"}

# Route for the main page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "models": available_models, "selected_model": ""}
    )


@app.post("/", response_class=HTMLResponse)
async def process_form(
    request: Request,
    text: str = Form(...),
    model: str = Form(...)
):
    if not text.strip():
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "models": available_models,
                "error": "Input text cannot be empty or whitespace.",
                "token_label_pairs": []
            }
        )

    # 🔽 Load and process the text
    result = process_text(text, model)

    # 🔽 Map internal model name to display name
    display_models = factory.get_display_models()
    official_model_name = next(
        (m["display_name"] for m in display_models if m["internal_name"] == model),
        model
    )

    if "error" in result:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "models": available_models,
                "error": result["error"],
                "token_label_pairs": [],
                "selected_model": official_model_name,
                "input_text": text
            }
        )

    token_label_pairs = result.get("token_label_pairs", [])
    if not token_label_pairs:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "models": available_models,
                "error": "No tokens processed from the input text.",
                "token_label_pairs": [],
                "selected_model": official_model_name,
                "input_text": text
            }
        )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "models": available_models,
            "token_label_pairs": token_label_pairs,
            "selected_model": official_model_name,
            "input_text": text
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)