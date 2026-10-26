
import json
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
    with (DATA_DIR / filename).open(
        "r", encoding="utf-8"
    ) as file:
        return json.load(file)


def save_json(filename, data):
    with (RESULT_DIR / filename).open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def compare_resume(actual, expected):
    return (
        actual.name == expected["name"]
        and set(actual.skills) == set(expected["skills"])
        and actual.years_experience == expected["years_experience"]
    )


def main():
    resumes = load_json("resumes.json")
    results = []

    for resume in resumes:
        expected = resume["expected"]

        try:
            validated = ResumeData.model_validate(expected)
            exact_match = compare_resume(validated, expected)

            results.append({
                "resume_id": resume["id"],
                "mode": "LOCAL_EXPECTED_DATA_VALIDATION",
                "valid_json_schema": True,
                "exact_match": exact_match,
                "invalid": False,
                "data": validated.model_dump(),
            })

        except (ValidationError, KeyError, TypeError) as error:
            results.append({
                "resume_id": resume.get("id"),
                "mode": "LOCAL_EXPECTED_DATA_VALIDATION",
                "valid_json_schema": False,
                "exact_match": False,
                "invalid": True,
                "error": str(error),
            })

    save_json("local_resume_extraction_results.json", results)

    total = len(results)
    valid = sum(item["valid_json_schema"] for item in results)
    invalid = total - valid
    exact = sum(item["exact_match"] for item in results)

    summary = {
        "mode": "LOCAL_EXPECTED_DATA_VALIDATION",
        "api_used": False,
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "exact_matches": exact,
        "note": (
            "Validates expected dataset only; does not measure "
            "Gemini extraction accuracy or JSON-mode performance."
        ),
    }

    save_json("local_resume_extraction_summary.json", summary)

    print("\nRESUME EXTRACTION LOCAL VALIDATION")
    print(f"Total resumes: {total}")
    print(f"Valid: {valid}/{total}")
    print(f"Invalid: {invalid}/{total}")
    print(f"Expected-data matches: {exact}/{total}")
    print("Gemini API calls: 0")
    print("Results saved:", RESULT_DIR)


if __name__ == "__main__":
    main()
