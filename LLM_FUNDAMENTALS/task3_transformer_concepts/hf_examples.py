
from transformers import pipeline


def main():
    print("Loading BERT fill-mask model...")

    fill_mask = pipeline(
        "fill-mask",
        model="google-bert/bert-base-uncased"
    )

    bert_result = fill_mask(
        "The capital of France is [MASK]."
    )

    print("\nBERT fill-mask results:")
    for item in bert_result[:3]:
        print(
            f"{item['token_str'].strip()}: "
            f"{item['score']:.4f}"
        )

    print("\nLoading GPT-2 text-generation model...")

    generator = pipeline(
        "text-generation",
        model="openai-community/gpt2"
    )

    gpt_result = generator(
        "Artificial intelligence can help students",
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=50256
    )

    print("\nGPT-2 text generation:")
    print(gpt_result[0]["generated_text"])


if __name__ == "__main__":
    main()
