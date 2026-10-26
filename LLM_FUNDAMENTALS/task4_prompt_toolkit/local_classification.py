
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROMPT_DIR = BASE_DIR / "prompts"
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

VERSIONS = [
    "v1_basic",
    "v2_structured",
    "v3_few_shot",
]

VALID_CATEGORIES = {"Billing", "Technical", "Account"}
VALID_PRIORITIES = {"Low", "Medium", "High"}


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def local_baseline(ticket):
    """
    Simple keyword baseline for testing evaluation logic.
    This is NOT an LLM prediction.
    """
    text = ticket["ticket"].lower()

    if any(word in text for word in [
        "invoice", "payment", "refund", "charged",
        "billing", "subscription", "price"
    ]):
        category = "Billing"
    elif any(word in text for word in [
        "password", "login", "account", "profile",
        "locked", "sign in", "email address"
    ]):
        category = "Account"
    else:
        category = "Technical"

    if any(word in text for word in [
        "urgent", "immediately", "cannot access",
        "security breach", "lost access", "outage"
    ]):
        priority = "High"
    elif any(word in text for word in [
        "slow", "soon", "intermittent", "delayed"
    ]):
        priority = "Medium"
    else:
        priority = "Low"

    return {
        "category": category,
        "priority": priority,
    }


def main():
    tickets = load_json(DATA_DIR / "golden_tickets.json")
    results = []
    summary = {}

    for version in VERSIONS:
        prompt_file = PROMPT_DIR / f"{version}.txt"

        if not prompt_file.exists():
            print(f"Missing prompt file: {prompt_file}")
            continue

        prompt_text = prompt_file.read_text(encoding="utf-8")

        if "{ticket}" not in prompt_text:
            print(f"Warning: {version} has no ticket placeholder")

        correct = 0

        for ticket in tickets:
            prediction = local_baseline(ticket)

            category_valid = (
                prediction["category"] in VALID_CATEGORIES
            )
            priority_valid = (
                prediction["priority"] in VALID_PRIORITIES
            )

            matched = (
                prediction["category"] == ticket["category"]
                and prediction["priority"] == ticket["priority"]
            )

            correct += int(matched)

            results.append({
                "evaluation_mode": "LOCAL_KEYWORD_BASELINE",
                "version": version,
                "ticket_id": ticket["id"],
                "expected_category": ticket["category"],
                "expected_priority": ticket["priority"],
                "predicted_category": prediction["category"],
                "predicted_priority": prediction["priority"],
                "valid_output": category_valid and priority_valid,
                "correct": matched,
            })

        summary[version] = {
            "mode": "LOCAL_KEYWORD_BASELINE_NOT_LLM",
            "correct": correct,
            "total": len(tickets),
            "accuracy_percent": round(
                correct / len(tickets) * 100, 2
            ) if tickets else 0,
        }

    save_json(RESULT_DIR / "local_classification_results.json", results)
    save_json(RESULT_DIR / "local_classification_summary.json", summary)

    print("\nLOCAL BASELINE RESULTS")
    for version, metrics in summary.items():
        print(
            f"{version}: {metrics['correct']}/{metrics['total']} "
            f"({metrics['accuracy_percent']}%)"
        )

    print("\nImportant: these are keyword-baseline results.")
    print("They do not measure prompt versions or Gemini accuracy.")
    print("Results saved in:", RESULT_DIR)


if __name__ == "__main__":
    main()
