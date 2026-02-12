from fastapi import FastAPI
from pydantic import BaseModel
from src.essay_aggregator import analyze_essay

app = FastAPI(
    title="Binary Dyslexia Detection Service",
    version="1.0"
)

class EssayRequest(BaseModel):
    essay: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_dyslexia(req: EssayRequest):
   
   

    result = analyze_essay(req.essay)
    return result