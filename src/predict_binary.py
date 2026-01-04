import sys
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

MODEL_PATH = "models/xlmr_sinhala_dyslexia_binary.pt"
MODEL_NAME = "xlm-roberta-base"
LABELS = {0: "Non-Dyslexic", 1: "Dyslexic"}

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict_binary.py <essay.txt>")
        return

    # Read essay
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        text = f.read()

    # Load tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

    #Rebuild model architecture EXACTLY
    model = XLMRobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    # Load trained weights (state_dict)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    model.eval()

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    print(f"Prediction: {LABELS[pred]}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()
