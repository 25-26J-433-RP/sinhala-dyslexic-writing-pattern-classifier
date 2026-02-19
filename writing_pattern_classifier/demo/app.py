import sys
import os
import gradio as gr

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from writing_pattern_classifier.src.v1.pipeline import analyze_essay


def analyze_text(text):
    if not text.strip():
        return "Please enter text.", ""

    sentences = [s.strip() for s in text.split("\n") if s.strip()]
    result = analyze_essay(sentences)

    profile = result["essay_profile"]
    sentence_analysis = result["sentence_analysis"]

    return str(profile), str(sentence_analysis)


iface = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(
        lines=15,
        placeholder="Paste Sinhala essay here (separate sentences by new line)..."
    ),
    outputs=[
        gr.Textbox(label="Essay Profile"),
        gr.Textbox(label="Sentence Analysis")
    ],
    title="Sinhala Dyslexic Writing Pattern Analyzer",
    description="Analyzes Sinhala essays for dyslexic writing patterns."
)

if __name__ == "__main__":
    iface.launch()
