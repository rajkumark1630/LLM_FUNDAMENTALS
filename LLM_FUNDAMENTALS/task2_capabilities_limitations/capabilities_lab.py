
import os
import csv
import json
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY missing. Add it to your .env file."
    )

client = genai.Client(api_key=API_KEY)

OUTPUT_FILE = "task2_capabilities_limitations/results.csv"


def ask_gemini(prompt):
    """Call Gemini and return its actual text response."""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    return response.text or ""


tests = [
    {
        "test": "Summarisation",
        "prompt": (
            "Summarise in two sentences: "
            "A company introduced a support chatbot. "
            "It answers common questions around the clock. "
            "Complex billing complaints are sent to human agents."
        ),
        "rating": 0,
        "notes": "Rate factual accuracy and completeness from 1 to 5.",
    },
    {
        "test": "Classification",
        "prompt": (
            "Classify this ticket as Billing, Technical, or Account. "
            "Ticket: I was charged twice for my subscription. "
            "Return only the category."
        ),
        "rating": 0,
        "notes": "Expected category: Billing.",
    },
    {
        "test": "Extraction",
        "prompt": (
            'Extract name and years_experience as JSON: '
            '"Priya has worked as a Python developer for 4 years." '
            'Use {"name": "...", "years_experience": 0}.'
        ),
        "rating": 0,
        "notes": "Expected name Priya and 4 years.",
    },
    {
        "test": "Translation",
        "prompt": (
            "Translate into Tamil: "
            "'Your account password has been reset successfully.'"
        ),
        "rating": 0,
        "notes": "Manually verify translation meaning.",
    },
    {
        "test": "Code generation",
        "prompt": (
            "Write a Python function that returns the sum of "
            "a list of numbers. Include one test."
        ),
        "rating": 0,
        "notes": "Review and run generated code.",
    },
    {
        "test": "Knowledge cutoff",
        "prompt": (
            "What major event happened in the world yesterday? "
            "Give the date and a source. If uncertain, say so."
        ),
        "rating": 0,
        "notes": (
            "Compare with reliable news sources. "
            "This test alone does not establish an exact cutoff."
        ),
    },
    {
        "test": "Hallucination baseline",
        "prompt": (
            "Summarise the findings of the fictional paper "
            "'Quantum Clouds and Human Memory', supposedly written "
            "by Dr Alex Example in 2025. Do not assume it exists."
        ),
        "rating": 0,
        "notes": "Check whether the model invents paper details.",
    },
    {
        "test": "Hallucination with safeguard",
        "prompt": (
            "Is there a verified paper titled 'Quantum Clouds and "
            "Human Memory' by Dr Alex Example? Say 'I don't know' "
            "if you cannot verify it. Do not invent citations."
        ),
        "rating": 0,
        "notes": "Compare with the baseline response.",
    },
    {
        "test": "Context retrieval",
        "prompt": (
            "Read this text and report the secret reference code. "
            + ("This is ordinary filler text. " * 100)
            + "The reference code is BLUE-4821. "
            + ("More ordinary filler text. " * 100)
            + "Return only the reference code."
        ),
        "rating": 0,
        "notes": "Expected reference code: BLUE-4821.",
    },
    {
        "test": "Arithmetic",
        "prompt": (
            "A shop sells 3 notebooks for Rs 45 each and "
            "2 pens for Rs 20 each. You pay Rs 200. "
            "How much change should you receive? "
            "Show the calculation."
        ),
        "rating": 0,
        "notes": "Verify manually: change should be Rs 25.",
    },
    {
        "test": "Logic puzzle",
        "prompt": (
            "All roses in a garden are flowers. Some flowers are red. "
            "Can we conclude that some roses are red? Explain briefly."
        ),
        "rating": 0,
        "notes": "Correct answer: No, not from these premises alone.",
    },
]

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

fieldnames = [
    "timestamp_utc",
    "model",
    "test",
    "prompt",
    "result",
    "rating_1_to_5",
    "verification_notes",
]

file_exists = os.path.exists(OUTPUT_FILE)

with open(
    OUTPUT_FILE,
    "a",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    if not file_exists:
        writer.writeheader()

    for item in tests:
        print(f"\nRunning: {item['test']}")

        try:
            result = ask_gemini(item["prompt"])
            print(result)

            writer.writerow({
                "timestamp_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "model": MODEL,
                "test": item["test"],
                "prompt": item["prompt"],
                "result": result,
                "rating_1_to_5": "",
                "verification_notes": item["notes"],
            })

            file.flush()
            time.sleep(1)

        except Exception as error:
            print(f"API call failed: {error}")
            writer.writerow({
                "timestamp_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "model": MODEL,
                "test": item["test"],
                "prompt": item["prompt"],
                "result": f"ERROR: {error}",
                "rating_1_to_5": "",
                "verification_notes": (
                    "Retry after checking model, API access and quota."
                ),
            })
            file.flush()

print(f"\nResults saved to: {OUTPUT_FILE}")
print("Rate each successful answer from 1 to 5 after verification.")
