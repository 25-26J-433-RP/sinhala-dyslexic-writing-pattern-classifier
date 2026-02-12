"""
Gradio demo application for Sinhala Dyslexia Binary Essay Classifier.

Responsibility:
- Provide a simple web-based interface
- Accept Sinhala essays as input
- Display essay-level dyslexia prediction

This file contains NO ML logic.
It only calls the analysis pipeline defined in src/.
"""

import gradio as gr
from src.essay_aggregator import analyze_essay


def detect_dyslexia(essay_text):
    """
    Wrapper function for Gradio UI.

    Args:
        essay_text (str): Sinhala essay entered by the user

    Returns:
        str: Human-readable prediction summary
    """
    result = analyze_essay(essay_text)

    if "error" in result:
        return result["error"]

    return (
        f"Prediction: {result['essay_label']}\n"
        f"Dyslexic sentences: {result['dyslexic_sentences']}/"
        f"{result['total_sentences']}\n"
        f"Dyslexia confidence: {result['confidence']}"
    )


# ------------------------------------------------------------
# Gradio Interface Definition
# ------------------------------------------------------------
demo = gr.Interface(
    fn=detect_dyslexia,
    inputs=gr.Textbox(lines=10, label="Paste Sinhala Essay"),
    outputs=gr.Textbox(label="Prediction"),
    title="Sinhala Dyslexia Binary Essay Classifier",
    description="Binary dyslexia detection for Sinhala essays (research prototype)."
)


# ------------------------------------------------------------
# Application Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    demo.launch()
