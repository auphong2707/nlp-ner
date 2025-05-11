from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
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
    if model not in get_model_keys():
        return {"error": "Invalid model selected"}
    predict_fn = load_model(model)
    return {"entities": predict_fn(text)}
