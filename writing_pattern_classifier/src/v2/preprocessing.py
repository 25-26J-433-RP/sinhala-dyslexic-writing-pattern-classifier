import regex


def extract_structured_features(text):
    features = {}

    features["char_length"] = len(text)
    features["word_count"] = len(text.split())

    graphemes = regex.findall(r"\X", text)

    repeated = sum(
        1 for i in range(1, len(graphemes))
        if graphemes[i] == graphemes[i - 1]
    )

    features["repeated_graphemes"] = repeated

    return features
