import gradio as gr
from src.router import analyze_essay


def format_report(result):

    output = []
    output.append("==============================")
    output.append("Enhanced Writing Pattern Diagnostic Report")
    output.append("------------------------------")
    output.append(f"Dominant Pattern: {result['dominant']}")
    output.append(f"Severity Level: {result['severity']}")
    output.append(f"Risk Level: {result['risk_level']}")
    output.append(f"Risk Score: {result['risk_score']:.2f}%\n")

    output.append("Explanation:")
    output.append(result["explanation"] + "\n")

    output.append("Pattern Distribution:")
    for k, v in result["distribution"].items():
        output.append(f"{k}: {v:.3f}")

    output.append("\nPattern Density (% of essay strongly affected):")
    for k, v in result["pattern_density"].items():
        output.append(f"{k}: {v:.1f}%")

    output.append("\nStrong Pattern Sentences:")
    for label, sentences in result["pattern_sentence_examples"].items():
        if sentences:
            output.append(f"\n{label} Pattern Sentences:")
            for s in sentences[:3]:
                output.append(f"- {s}")

    output.append("==============================")

    return "\n".join(output)


def analyze(essay_text):
    result = analyze_essay(essay_text)
    return format_report(result)


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(lines=10, placeholder="Paste Sinhala essay here..."),
    outputs=gr.Textbox(lines=20),
    title="Sinhala Dyslexic Writing Pattern Classifier (V2)",
    description="ML-based essay-level writing pattern analysis."
)

if __name__ == "__main__":
    demo.launch()
