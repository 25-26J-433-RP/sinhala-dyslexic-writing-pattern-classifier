from fastapi import FastAPI
from pydantic import BaseModel

from binary_dyslexia_detector.src.essay_aggregator import analyze_essay
# Later:
# from writing_pattern_classifier.src.pattern_pipeline import analyze_patterns

app = FastAPI(
    title="Sinhala Dyslexia Pattern Analysis Service",
    version="2.0"
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

@app.post("/analyze")
def full_analysis(req: EssayRequest):
    binary_result = analyze_essay(req.essay)

    # pattern_result = analyze_patterns(req.essay)

    return {
        "binary": binary_result,
        # "patterns": pattern_result
    }
