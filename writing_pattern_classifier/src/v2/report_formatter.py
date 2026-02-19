def print_diagnostic_report(result):

    print("\n==============================")
    print("Enhanced Writing Pattern Diagnostic Report")
    print("------------------------------")
    print("Dominant Pattern:", result["dominant"])
    print("Severity Level:", result["severity"])
    print("Risk Level:", result["risk_level"])
    print(f"Risk Score: {result['risk_score']:.2f}%")

    print("\nExplanation:")
    print(result["explanation"])

    print("\nPattern Distribution:")
    for k, v in result["distribution"].items():
        print(f"{k}: {v:.3f}")

    print("\nPattern Density (% of essay strongly affected):")
    for k, v in result["pattern_density"].items():
        print(f"{k}: {v:.1f}%")

    print("\nStrong Pattern Sentences:")

    for label, sentences in result["pattern_sentence_examples"].items():
        if sentences:
            print(f"\n{label} Pattern Sentences:")
            for s in sentences[:3]:
                print("-", s)

    print("==============================")
