
import tiktoken
import pandas as pd


texts = {
    "Sentence": (
        "Large language models generate text using tokens."
    ),
    "Paragraph": (
        "Large language models process text as tokens. "
        "They use attention to understand relationships "
        "between words and generate responses."
    ),
    "Page": (
        "Language models learn statistical patterns from data. "
        "Tokenization converts text into smaller units. "
        "Transformers use self-attention to process context. "
    ) * 35,
    "JSON": (
        '{"name":"Arun","skills":["Python","SQL"],'
        '"years_experience":3}'
    ),
    "Tamil": "வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?",
}

encoding = tiktoken.get_encoding("cl100k_base")
rows = []

for text_type, text in texts.items():
    tokens = encoding.encode(text)
    words = len(text.split())
    token_count = len(tokens)

    rows.append({
        "Text Type": text_type,
        "Words": words,
        "Tokens": token_count,
        "Words per Token": (
            round(words / token_count, 3)
            if token_count else 0
        ),
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))

for context_size in (8000, 128000):
    print(f"\nContext window: {context_size:,} tokens")
    for row in rows:
        fits = row["Tokens"] <= context_size
        print(f"{row['Text Type']}: {'Fits' if fits else 'Does not fit'}")
