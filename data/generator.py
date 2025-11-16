import json
import random

# Define Sinhala substitution rules to simulate dyslexic patterns
VOWEL_CONFUSIONS = [("ි", "ී"), ("ෙ", "ේ"), ("ු", "ූ")]
CONSONANT_SWAPS = [("ක", "ග"), ("ත", "ද"), ("න", "ම"), ("ර", "ල")]
SUFFIX_DROPS = ["ට", "ටේ"]

# Some simple base sentences (clean)
base_sentences = [
    "මම පාසලට ගියා",
    "ඇය ගුරුතුමියට කතා කළා",
    "අපි බස් එකට ගියා",
    "ඔහු පොත කියවා",
    "ඔවුන් ගෙදරට ගියා"
]

def inject_error(word):
    """Randomly add a dyslexic error to a Sinhala word"""
    choice = random.choice(["vowel", "consonant", "suffix", "none"])

    if choice == "vowel":
        src, tgt = random.choice(VOWEL_CONFUSIONS)
        return word.replace(src, tgt, 1), "vowel_confusion"

    elif choice == "consonant":
        src, tgt = random.choice(CONSONANT_SWAPS)
        return word.replace(src, tgt, 1), "graphemic_swap"

    elif choice == "suffix" and word.endswith("ට"):
        return word[:-1], "suffix_omission"

    else:
        return word, "O"

def generate_samples(n=50):
    dataset = []

    for _ in range(n):
        text = random.choice(base_sentences)
        tokens = text.split()
        new_tokens = []
        labels = []

        for token in tokens:
            new_word, label = inject_error(token)
            new_tokens.append(new_word)
            labels.append(label)

        dataset.append({
            "text": " ".join(new_tokens),
            "tokens": new_tokens,
            "labels": labels
        })

    return dataset

if __name__ == "__main__":
    data = generate_samples(100)
    with open("sinhala_dyslexia_samples.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("✅ Generated synthetic Sinhala dyslexic dataset → sinhala_dyslexia_samples.json")
