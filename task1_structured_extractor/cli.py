import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from task1_structured_extractor.extractor import extract_resume
from task1_structured_extractor.gemini_provider import generate_resume_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured resume data into validated JSON."
    )
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "sample_data" / "resume.txt"),
        help="Path to the input resume text file.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "output" / "resume_profile.json"),
        help="Path where the JSON result will be saved.",
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "live"],
        default="demo",
        help="Use a fixed sample response or call the Gemini API.",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        parser.error(f"Input file not found: {input_path}")

    resume_text = input_path.read_text(encoding="utf-8-sig")

    if args.mode == "demo":
        response_path = (
            PROJECT_ROOT / "sample_data" / "demo_resume_response.json"
        )
        json_response = response_path.read_text(encoding="utf-8-sig")
        profile = extract_resume(resume_text, lambda prompt: json_response)
    else:
        profile = extract_resume(resume_text, generate_resume_json)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Mode: {args.mode}")
    print(f"Validated resume: {profile.full_name}")
    print(f"Output saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
