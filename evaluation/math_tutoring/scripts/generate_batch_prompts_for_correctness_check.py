#!/usr/bin/env python3
"""
Generate batch prompts for correctness evaluation in math tutoring task.

This script prepares correctness evaluation prompts based on extracted answers.
"""

import os
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


def load_extracted_answers(file_name: str) -> Dict:
    """Load extracted answers from evaluation outputs."""
    answers_path = Path(__file__).parent.parent / "evaluation_outputs" / "extracted_answer" / f"{file_name}.json"

    if not answers_path.exists():
        raise FileNotFoundError(f"Extracted answers file not found: {answers_path}\n"
                              f"Please run answer extraction first.")

    with open(answers_path, 'r') as f:
        data = json.load(f)
        if "answers" in data:
            return data["answers"]
        else:
            raise ValueError(f"Invalid format in {answers_path}: missing 'answers' key")

def load_annotations(annotation_id: str) -> List[Dict]:
    """Load annotation data from SimulatorArena data folder."""
    annotations_path = Path(__file__).parent.parent.parent.parent / "data" / "math_tutoring_annotations.json"

    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    with open(annotations_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(
        description="Generate batch prompts for correctness evaluation in math tutoring."
    )

    # Required arguments
    parser.add_argument(
        "--file_name",
        type=str,
        required=True,
        help="Name of the simulation output file (without .json extension)"
    )

    # Optional arguments
    parser.add_argument(
        "--annotation_id",
        type=str,
        default="math_tutoring_annotations",
        help="Annotation dataset ID (default: math_tutoring_annotations)"
    )
    parser.add_argument(
        "--evaluator_model",
        type=str,
        default="gpt-5-mini",
        help="Model to use for evaluation (default: gpt-5-mini)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=5000,
        help="Maximum tokens for response (default: 5000)"
    )

    args = parser.parse_args()

    print(f"Generating correctness check prompts for: {args.file_name}")
    print(f"Using evaluator model: {args.evaluator_model}")

    # Load prompt template
    prompt_template_path = Path(__file__).parent.parent / "prompts" / "check_correctness.txt"
    if not prompt_template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_template_path}")

    with open(prompt_template_path, 'r') as f:
        prompt_template = f.read()

    # Load annotations
    annotations = load_annotations(args.annotation_id)
    annotation_lookup = {ann["problem_id"]: ann for ann in annotations}

    # Load extracted answers
    extracted_answers = load_extracted_answers(args.file_name)

    # Check for existing evaluations to avoid re-processing
    # Correctness results are stored in the same file as extracted answers
    output_file = Path(__file__).parent.parent / "evaluation_outputs" / "extracted_answer" / f"{args.file_name}.json"
    existing_evaluations = {}
    if output_file.exists():
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
            if "answers" in existing_data:
                # Check if any answers already have correctness data
                for model in existing_data["answers"]:
                    for problem_id in existing_data["answers"][model]:
                        for user_key in existing_data["answers"][model][problem_id]:
                            if "correctness" in existing_data["answers"][model][problem_id][user_key]:
                                if model not in existing_evaluations:
                                    existing_evaluations[model] = {}
                                if problem_id not in existing_evaluations[model]:
                                    existing_evaluations[model][problem_id] = {}
                                existing_evaluations[model][problem_id][user_key] = True

    # Prepare batch prompts
    batch_prompts = []
    keys = []
    skipped = 0
    missing_answer = 0

    for model_name in tqdm(extracted_answers, desc="Processing models"):
        for problem_id in extracted_answers[model_name]:
            problem_id_str = str(problem_id)

            # Get the correct answer from annotations
            problem_id_int = int(problem_id) if isinstance(problem_id, str) else problem_id
            if problem_id_int not in annotation_lookup:
                print(f"Warning: No annotation found for problem {problem_id}")
                continue

            annotation = annotation_lookup[problem_id_int]
            correct_answer = annotation["problem_1_gold_final_answer"]
            math_problem = annotation["question"]

            for user_key, answer_data in extracted_answers[model_name][problem_id_str].items():
                # Skip if already evaluated
                if (existing_evaluations and model_name in existing_evaluations and
                    problem_id_str in existing_evaluations[model_name] and
                    user_key in existing_evaluations[model_name][problem_id_str]):
                    skipped += 1
                    continue

                # Get the extracted answer
                if "extracted_answer" not in answer_data:
                    print(f"Warning: No extracted answer for {model_name}/{problem_id}/{user_key}")
                    missing_answer += 1
                    continue

                student_answer = answer_data["extracted_answer"]

                # Skip if answer extraction failed
                if student_answer == "Error":
                    missing_answer += 1
                    continue

                # Create correctness evaluation prompt
                prompt = prompt_template.format(
                    question=math_problem,
                    correct_answer=correct_answer,
                    student_answer=student_answer
                )

                batch_prompts.append({
                    "custom_id": f"{model_name}|{problem_id_str}|{user_key}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.evaluator_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 1.0,
                        "max_completion_tokens": args.max_tokens
                    }
                })
                keys.append((model_name, problem_id_str, user_key))

    print(f"\nGenerated {len(batch_prompts)} prompts ({skipped} skipped as already evaluated, {missing_answer} missing/error answers)")

    if len(batch_prompts) == 0:
        print("No new prompts to process. Exiting.")
        return

    # Save batch prompts in JSONL format for OpenAI batch API
    batch_dir = Path(__file__).parent.parent / "batch_prompts" / "extracted_answer"
    batch_dir.mkdir(parents=True, exist_ok=True)

    batch_file = batch_dir / f"{args.file_name}_correctness.jsonl"

    # Ensure parent directory exists (handles subdirectories in file_name like "gpt-5-mini/...")
    batch_file.parent.mkdir(parents=True, exist_ok=True)

    with open(batch_file, 'w') as f:
        for prompt in batch_prompts:
            f.write(json.dumps(prompt) + '\n')

    print(f"Batch prompts saved to: {batch_file}")

    # Save keys for later mapping
    keys_file = batch_dir / f"{args.file_name}_correctness_keys.json"
    keys_file.parent.mkdir(parents=True, exist_ok=True)
    with open(keys_file, 'w') as f:
        json.dump({"keys": [list(k) for k in keys]}, f, indent=2)

    print(f"Keys saved to: {keys_file}")

    return batch_file

if __name__ == "__main__":
    main()
