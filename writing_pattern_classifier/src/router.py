USE_VERSION = "v2"


def analyze_essay(essay_text):
    if USE_VERSION == "v1":
        from .v1.pipeline import analyze_essay as analyze
    else:
        from .v2.pipeline import analyze_essay as analyze



    return analyze(essay_text)
