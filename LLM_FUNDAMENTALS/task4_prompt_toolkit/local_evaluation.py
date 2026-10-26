
import json
from pathlib import Path
from collections import Counter

from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


class ResumeData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    skills: list[str]
    years_experience: int = Field(ge=0)


def load_json(filename):
    path = DATA_DIR / filename
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(filename, data):
    path = RESULT_DIR / filename
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def validate_tickets():
    tickets = load_json("golden_tickets.json")
    categories = {"Billing", "Technical", "Account"}
    priorities = {"Low", "Medium", "High"}

    errors = []
    category_counts = Counter()
    priority_counts = Counter()

    for ticket in tickets:
        if ticket.get("category") not in categories:
            errors.append(
                f"Ticket {ticket.get('id')}: invalid category"
            )

        if ticket.get("priority") not in priorities:
            errors.append(
                f"Ticket {ticket.get('id')}: invalid priority"
            )

        if not ticket.get("ticket", "").strip():
            errors.append(
                f"Ticket {ticket.get('id')}: empty ticket text"
            )

        category_counts[ticket.get("category")] += 1
        priority_counts[ticket.get("priority")] += 1

    result = {
        "mode": "LOCAL_ONLY",
        "api_used": False,
        "total_tickets": len(tickets),
        "valid_tickets": len(tickets) - len(errors),
        "errors": errors,
        "category_distribution": dict(category_counts),
        "priority_distribution": dict(priority_counts),
    }

    save_json("local_ticket_validation.json", result)

    print("\n--- TICKET VALIDATION ---")
    print(f"Total tickets: {len(tickets)}")
    print(f"Valid tickets: {len(tickets) - len(errors)}")
    print(f"Errors: {len(errors)}")
    print("Categories:", dict(category_counts))
    print("Priorities:", dict(priority_counts))


def validate_resumes():
    resumes = load_json("resumes.json")
    rows = []

    for resume in resumes:
        row = {
            "resume_id": resume.get("id"),
            "valid": False,
            "errors": [],
        }

        try:
            expected = ResumeData.model_validate(
                resume["expected"]
            )
            row["valid"] = True
            row["validated_expected"] = expected.model_dump()

        except Exception as error:
            row["errors"] = str(error).splitlines()

        rows.append(row)

    valid_count = sum(row["valid"] for row in rows)

    save_json("local_resume_validation.json", rows)

    print("\n--- RESUME VALIDATION ---")
    print(f"Total resumes: {len(resumes)}")
    print(f"Valid expected JSON: {valid_count}/{len(resumes)}")
    print(f"Invalid: {len(resumes) - valid_count}")


def validate_prompt_files():
    prompt_dir = BASE_DIR / "prompts"

    filenames = [
        "v1_basic.txt",
        "v2_structured.txt",
        "v3_few_shot.txt",
    ]

    rows = []

    for filename in filenames:
        path = prompt_dir / filename

        row = {
            "file": filename,
            "exists": path.exists(),
            "contains_ticket_placeholder": False,
            "characters": 0,
        }

        if path.exists():
            content = path.read_text(encoding="utf-8")
            row["contains_ticket_placeholder"] = (
                "{ticket}" in content
            )
            row["characters"] = len(content)

        rows.append(row)

    save_json("local_prompt_validation.json", rows)

    print("\n--- PROMPT FILES ---")
    for row in rows:
        print(
            f"{row['file']}: exists={row['exists']}, "
            f"placeholder={row['contains_ticket_placeholder']}, "
            f"characters={row['characters']}"
        )


def validate_json_parser():
    example = {
        "name": "Test User",
        "skills": ["Python", "SQL"],
        "years_experience": 2,
    }

    encoded = json.dumps(example)
    decoded = json.loads(encoded)
    validated = ResumeData.model_validate(decoded)

    result = {
        "json_round_trip": decoded == example,
        "pydantic_validation": validated.model_dump() == example,
        "api_used": False,
    }

    save_json("local_parser_test.json", result)

    print("\n--- JSON AND PYDANTIC TEST ---")
    print("JSON round trip:", result["json_round_trip"])
    print("Pydantic validation:", result["pydantic_validation"])


def main():
    print("TASK 4 LOCAL EVALUATION")
    print("Gemini API calls: 0")

    validate_tickets()
    validate_resumes()
    validate_prompt_files()
    validate_json_parser()

    print("\nLocal evaluation finished.")
    print("Results folder:", RESULT_DIR)
    print(
        "\nNote: This validates the dataset and local components. "
        "It does not measure actual LLM classification accuracy."
    )


if __name__ == "__main__":
    main()
