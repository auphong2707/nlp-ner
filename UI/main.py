from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from model_loader import load_model, get_model_keys

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_keys": get_model_keys()
    })

@app.post("/ner/")
def run_ner(text: str = Form(...), model: str = Form(...)):
    print(text)
    if model not in get_model_keys():
        return {"error": "Invalid model selected"}

    predict_fn = load_model(model)
    output = predict_fn(text)

    if not isinstance(output, dict) or "tokens" not in output or "labels" not in output:
        return {"error": "Model did not return tokens and labels"}

    return {
        "tokens": output["tokens"],
        "labels": output["labels"]
    }

@app.post("/ner/")
def run_ner(text: str = Form(...), model: str = Form(...)):
    print(f"Received text: {text}, model: {model}")  # log the received data
    if model not in get_model_keys():
        return {"error": "Invalid model selected"}

    predict_fn = load_model(model)
    output = predict_fn(text)

    print(f"Model output: {output}")  # log the model output

    if not isinstance(output, dict) or "tokens" not in output or "labels" not in output:
        return {"error": "Model did not return tokens and labels"}

    return {
        "tokens": output["tokens"],
        "labels": output["labels"]
    }