from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from model_loader import load_model, get_model_keys

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Home endpoint that renders the template with the available model keys.
    """
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_keys": get_model_keys()
    })

@app.post("/ner/")
def run_ner(text: str = Form(...), model: str = Form(...)):
    """
    Endpoint to run Named Entity Recognition (NER) on the input text
    using the selected model.
    """
    print(f"Received text: {text}, model: {model}")  # log the received data

    # Check if the model exists in the model configuration
    if model not in get_model_keys():
        return {"error": "Invalid model selected"}

    # Load the model
    predict_fn = load_model(model)
    
    # Run prediction
    output = predict_fn(text)

    # Log the output for debugging
    print(f"Model output: {output}")  # log the model output

    # Check if the output contains tokens and labels
    if not isinstance(output, dict) or "tokens" not in output or "labels" not in output:
        return {"error": "Model did not return tokens and labels"}

    # Return the tokens and labels as the response
    return {
        "tokens": output["tokens"],
        "labels": output["labels"]
    }
