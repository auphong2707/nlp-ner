from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
import sys
import os

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
available_models = factory.get_available_models()

# Simple test route to confirm the server is responding
@app.get("/test")
async def test():
    return {"message": "Server is running!"}

# Function to process input text through the selected model
def process_text(text: str, model_name: str) -> dict:
    try:
        # Load the model and tokenizer
        loader = factory.get_loader(model_name)
        model, tokenizer = loader.load()

        # Tokenize the input text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True
        )

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
        else:  # For non-CRF models (e.g., T5ForTokenClassification)
            logits = outputs.logits  # Shape: (batch_size, sequence_length, num_labels)
            predicted_labels = torch.argmax(logits, dim=-1).cpu().numpy()[0]  # Take argmax for predictions

        # Decode tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu().tolist())

        # Get label names (assuming the model config has id2label)
        id2label = model.config.id2label
        predicted_labels_str = [id2label[label_id] for label_id in predicted_labels]

        # Post-process to get final labels (e.g., remove labels for special tokens)
        final_tokens = []
        final_labels = []
        for token, label in zip(tokens, predicted_labels_str):
            # Skip special tokens (e.g., '</s>', '<pad>', etc.)
            if token in tokenizer.all_special_tokens:
                continue
            final_tokens.append(token)
            final_labels.append(label)

        return {
            "tokens": final_tokens,
            "predicted_labels": predicted_labels_str[:len(final_tokens)],  # Align with final tokens
            "final_labels": final_labels
        }
    except Exception as e:
        return {"error": str(e)}

# Route for the main page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "models": available_models}
    )

@app.post("/", response_class=HTMLResponse)
async def process_form(
    request: Request,
    text: str = Form(...),
    model: str = Form(...)
):
    if not text:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "models": available_models,
                "error": "Please enter some text."
            }
        )

    # Process the text with the selected model
    result = process_text(text, model)

    if "error" in result:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "models": available_models,
                "error": result["error"]
            }
        )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "models": available_models,
            "tokens": result["tokens"],
            "predicted_labels": result["predicted_labels"],
            "final_labels": result["final_labels"],
            "selected_model": model,
            "input_text": text
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)