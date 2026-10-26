
import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROMPT_DIR = BASE_DIR / "prompts"
RESULT_DIR = BASE_DIR / "results"

for folder in (DATA_DIR, PROMPT_DIR, RESULT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
client = genai.Client(api_key=API_KEY) if API_KEY else None


class ResumeData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    skills: list[str]
    years_experience: int = Field(ge=0)


def load_json(filename):
    with (DATA_DIR / filename).open(
        encoding="utf-8"
    ) as f:
        return json.load(f)


def save_json(filename, data):
    with (RESULT_DIR / filename).open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_json(text):
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    return json.loads(text)


def ask_gemini(prompt, temperature=0, top_p=1.0,
               json_mode=False):
    if client is None:
        raise RuntimeError("GEMINI_API_KEY missing in .env")

    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        response_mime_type=(
            "application/json" if json_mode else "text/plain"
        ),
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=config,
    )

    if not response.text:
        raise ValueError("Empty Gemini response")

    return response.text


def is_api_error(error):
    message = str(error).lower()
    return any(word in message for word in (
        "429", "quota", "resource_exhausted",
        "rate limit", "503", "500", "401", "403"
    ))


def run_classification():
    tickets = load_json("golden_tickets.json")
    versions = {
        "v1_basic": "v1_basic.txt",
        "v2_structured": "v2_structured.txt",
        "v3_few_shot": "v3_few_shot.txt",
    }

    rows = []
    summary = {}

    for version, filename in versions.items():
        path = PROMPT_DIR / filename

        if not path.exists():
            print(f"Missing prompt: {path}")
            continue

        template = path.read_text(encoding="utf-8")
        correct = 0
        successful = 0

        print(f"\nTesting {version}")

        for ticket in tickets:
            row = {
                "version": version,
                "ticket_id": ticket["id"],
                "expected_category": ticket["category"],
                "expected_priority": ticket["priority"],
                "predicted_category": None,
                "predicted_priority": None,
                "correct": None,
                "status": "",
            }

            try:
                prompt = template.replace(
                    "{ticket}", ticket["ticket"]
                )
                output = ask_gemini(
                    prompt, temperature=0, json_mode=True
                )
                answer = parse_json(output)

                category = answer.get("category")
                priority = answer.get("priority")

                if category not in {
                    "Billing", "Technical", "Account"
                }:
                    raise ValueError("Invalid category")

                if priority not in {"Low", "Medium", "High"}:
                    raise ValueError("Invalid priority")

                matched = (
                    category == ticket["category"]
                    and priority == ticket["priority"]
                )

                correct += int(matched)
                successful += 1

                row.update({
                    "predicted_category": category,
                    "predicted_priority": priority,
                    "correct": matched,
                    "status": "SUCCESS",
                    "output": output,
                })

            except Exception as error:
                row["status"] = (
                    "API_ERROR" if is_api_error(error)
                    else "INVALID_OUTPUT"
                )
                row["output"] = str(error)
                print(f"Ticket {ticket['id']}: {error}")

            rows.append(row)

            if row["status"] == "API_ERROR":
                print("API quota/error. Stopping classification.")
                break

        accuracy = (
            round(correct / successful * 100, 2)
            if successful else None
        )

        summary[version] = {
            "correct": correct,
            "successful": successful,
            "accuracy_percent": accuracy,
        }

        print(f"{version}: {correct}/{successful}; accuracy={accuracy}%")

        if any(
            r["version"] == version and r["status"] == "API_ERROR"
            for r in rows
        ):
            break

    save_json("classification_results.json", rows)
    save_json("classification_summary.json", summary)


def run_reasoning():
    problems = [
        ("12 apples - 5 apples = ?", 7),
        ("20 passengers; 8 leave; 3 board. Total?", 15),
        ("3 notebooks cost 40 rupees each. Total?", 120),
        ("15 pages daily for 4 days. Total?", 60),
        ("24 cookies shared among 6 children. Each?", 4),
    ]

    rows = []

    for question, expected in problems:
        for style in ("plain", "step_by_step"):
            instruction = (
                "Return only the numerical answer."
                if style == "plain"
                else "Solve carefully and end with FINAL: number."
            )

            row = {
                "style": style,
                "question": question,
                "expected": expected,
                "predicted": None,
                "correct": None,
                "status": "",
            }

            try:
                import re

                output = ask_gemini(
                    f"{instruction}\n{question}", temperature=0
                )

                match = re.search(
                    r"FINAL:\s*(-?\d+(?:\.\d+)?)",
                    output, re.IGNORECASE
                )
                if not match:
                    match = re.search(r"-?\d+(?:\.\d+)?", output)

                if not match:
                    raise ValueError("No numeric answer")

                predicted = float(
                    match.group(1) if match.lastindex
                    else match.group()
                )

                row.update({
                    "predicted": predicted,
                    "correct": predicted == expected,
                    "status": "SUCCESS",
                    "output": output,
                })

            except Exception as error:
                row["status"] = (
                    "API_ERROR" if is_api_error(error) else "ERROR"
                )
                row["output"] = str(error)

            rows.append(row)

            if row["status"] == "API_ERROR":
                save_json("cot_results.json", rows)
                print("Reasoning stopped due to API error.")
                return

    save_json("cot_results.json", rows)

    for style in ("plain", "step_by_step"):
        subset = [
            r for r in rows
            if r["style"] == style and r["status"] == "SUCCESS"
        ]
        score = sum(r["correct"] is True for r in subset)
        print(f"{style}: {score}/{len(subset)} correct")


def run_sampling():
    prompt = "Write a two-sentence story about a robot growing a plant."
    experiments = (
        [("temperature", t, 1.0) for t in (0, 0.7, 1.2)]
        + [("top_p", 0.7, p) for p in (0.3, 0.95)]
    )

    rows = []

    for name, temperature, top_p in experiments:
        for run in range(1, 6):
            row = {
                "experiment": name,
                "temperature": temperature,
                "top_p": top_p,
                "run": run,
                "status": "",
            }

            try:
                row["output"] = ask_gemini(
                    prompt, temperature=temperature, top_p=top_p
                )
                row["status"] = "SUCCESS"
            except Exception as error:
                row["status"] = (
                    "API_ERROR" if is_api_error(error) else "ERROR"
                )
                row["output"] = str(error)

            rows.append(row)

            if row["status"] == "API_ERROR":
                save_json("sampling_results.json", rows)
                print("Sampling stopped due to API quota/error.")
                return

    save_json("sampling_results.json", rows)
    print("Sampling results saved.")


def run_extraction():
    resumes = load_json("resumes.json")
    rows = []

    for resume in resumes:
        for mode in ("plain", "json_mode"):
            prompt = f"""
Extract name, skills (array of strings), and years_experience
(non-negative integer) from this resume.
Use 0 if experience is not stated. Do not invent facts.
Return JSON only.

Resume:
{resume['text']}
"""

            row = {
                "resume_id": resume["id"],
                "mode": mode,
                "expected": resume["expected"],
                "extracted": None,
                "valid": False,
                "exact_match": False,
                "status": "",
            }

            try:
                output = ask_gemini(
                    prompt, temperature=0,
                    json_mode=(mode == "json_mode")
                )
                extracted = ResumeData.model_validate(
                    parse_json(output)
                ).model_dump()

                expected = resume["expected"]
                exact = (
                    extracted["name"] == expected["name"]
                    and set(extracted["skills"]) == set(expected["skills"])
                    and extracted["years_experience"]
                    == expected["years_experience"]
                )

                row.update({
                    "extracted": extracted,
                    "valid": True,
                    "exact_match": exact,
                    "status": "SUCCESS",
                    "output": output,
                })

            except Exception as error:
                row["status"] = (
                    "API_ERROR" if is_api_error(error)
                    else "INVALID_OUTPUT"
                )
                row["error"] = str(error)

            rows.append(row)

            if row["status"] == "API_ERROR":
                save_json("extraction_results.json", rows)
                print("Extraction stopped due to API quota/error.")
                return

    save_json("extraction_results.json", rows)

    for mode in ("plain", "json_mode"):
        subset = [r for r in rows if r["mode"] == mode]
        successful = [r for r in subset if r["status"] == "SUCCESS"]
        invalid = sum(not r["valid"] for r in successful)
        exact = sum(r["exact_match"] for r in successful)
        print(
            f"{mode}: {invalid}/{len(successful)} invalid; "
            f"{exact}/{len(successful)} exact"
        )


def save_prompt_rewrites():
    rewrites = [
        {
            "bad": "Classify this ticket.",
            "improved": (
                "Classify as Billing, Technical, or Account. "
                "Assign Low, Medium, or High priority. Return JSON "
                "with category and priority. Use only ticket facts."
            ),
        },
        {
            "bad": "Extract resume details.",
            "improved": (
                "Return JSON with name (string), skills (string array), "
                "and years_experience (integer >= 0). Use 0 if unstated."
            ),
        },
        {
            "bad": "Reply nicely to the customer.",
            "improved": (
                "Write a professional support reply under 100 words. "
                "Use only supplied facts. Do not promise unconfirmed "
                "actions. Ask one clear question if information is missing."
            ),
        },
    ]
    save_json("prompt_rewrites.json", rewrites)


def run_local():
    tickets = load_json("golden_tickets.json")
    resumes = load_json("resumes.json")

    categories = {"Billing", "Technical", "Account"}
    priorities = {"Low", "Medium", "High"}
    errors = []

    for ticket in tickets:
        if ticket.get("category") not in categories:
            errors.append(f"Ticket {ticket.get('id')}: bad category")
        if ticket.get("priority") not in priorities:
            errors.append(f"Ticket {ticket.get('id')}: bad priority")
        if not ticket.get("ticket", "").strip():
            errors.append(f"Ticket {ticket.get('id')}: empty text")

    valid_resumes = 0
    resume_errors = []

    for resume in resumes:
        try:
            ResumeData.model_validate(resume["expected"])
            valid_resumes += 1
        except Exception as error:
            resume_errors.append(str(error))

    ResumeData.model_validate(parse_json(
        '{"name":"Test","skills":["Python"],"years_experience":2}'
    ))

    save_prompt_rewrites()

    summary = {
        "mode": "LOCAL_ONLY",
        "gemini_api_called": False,
        "tickets_loaded": len(tickets),
        "ticket_errors": errors,
        "resumes_loaded": len(resumes),
        "valid_expected_resumes": valid_resumes,
        "resume_errors": resume_errors,
        "json_parser": "PASS",
        "pydantic": "PASS",
    }

    save_json("local_validation_results.json", summary)

    print("\nLOCAL VALIDATION")
    print(f"Tickets: {len(tickets)}")
    print(f"Ticket errors: {len(errors)}")
    print(f"Resumes: {len(resumes)}")
    print(f"Valid expected resumes: {valid_resumes}/{len(resumes)}")
    print("JSON parser: PASS")
    print("Pydantic: PASS")
    print("Gemini API calls: 0")
    print(f"Results saved: {RESULT_DIR}")


def main():
    save_prompt_rewrites()

    if "--local" in sys.argv:
        run_local()
        return

    if not API_KEY:
        print("GEMINI_API_KEY missing in .env")
        print("Run: python task4_prompt_toolkit.py --local")
        return

    print(f"Using Gemini model: {MODEL}")

    for experiment in (
        run_classification,
        run_reasoning,
        run_sampling,
        run_extraction,
    ):
        try:
            experiment()
        except Exception as error:
            print(f"{experiment.__name__} failed: {error}")
            if is_api_error(error):
                print("Stopping because of Gemini API quota/error.")
                break

    print(f"Results saved: {RESULT_DIR}")


if __name__ == "__main__":
    main()
