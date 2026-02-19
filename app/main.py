from fastapi import FastAPI
from pydantic import BaseModel

from binary_dyslexia_detector.src.essay_aggregator import analyze_essay as analyze_binary
from writing_pattern_classifier.src.router import analyze_essay as analyze_patterns

app = FastAPI(
    title="Sinhala Dyslexia Pattern Analysis Service",
    version="3.0"
)

class EssayRequest(BaseModel):
    essay: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_dyslexia(req: EssayRequest):
    return analyze_binary(req.essay)


@app.post("/patterns")
def predict_patterns(req: EssayRequest):
    return analyze_patterns(req.essay)


@app.post("/analyze")
def full_analysis(req: EssayRequest):
    binary_result = analyze_binary(req.essay)
    pattern_result = analyze_patterns(req.essay)

    return {
        "binary": binary_result,
        "patterns": pattern_result
    }
