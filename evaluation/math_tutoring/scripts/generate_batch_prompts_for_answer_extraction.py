#!/usr/bin/env python3
"""
Generate batch prompts for answer extraction in math tutoring task.

This script prepares evaluation prompts for batch processing via OpenAI's API.
"""

import os
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


def load_simulation_data(file_name: str, annotation_id: str) -> Dict:
    """Load simulation output data from SimulatorArena."""
    simulation_path = Path(__file__).parent.parent.parent.parent / "simulation" / "output" / annotation_id / f"{file_name}.json"

    if not simulation_path.exists():
        raise FileNotFoundError(f"Simulation file not found: {simulation_path}")

    with open(simulation_path, 'r') as f:
        return json.load(f)

def load_terminated_conversations(file_name: str, annotation_id: str) -> Dict:
    """Load terminated conversation data."""
    terminated_path = Path(__file__).parent.parent.parent.parent / "simulation" / "terminated_conversations" / annotation_id / f"{file_name}.json"

    if not terminated_path.exists():
        print(f"Warning: Terminated conversations file not found: {terminated_path}")
        print("Using full conversations. Consider running termination detection first.")
        return {}

    with open(terminated_path, 'r') as f:
        return json.load(f)

def load_annotations(annotation_id: str) -> List[Dict]:
    """Load annotation data from SimulatorArena data folder."""
    annotations_path = Path(__file__).parent.parent.parent.parent / "data" / "math_tutoring_annotations.json"

    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    with open(annotations_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(
        description="Generate batch prompts for answer extraction in math tutoring."
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
        "--terminate_help",
        nargs='?',
        const=True,
        default=True,
        type=lambda x: str(x).lower() in ('true', '1', 'yes'),
        help="Use terminated conversation endpoints if available (default: True)"
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

    print(f"Generating answer extraction prompts for: {args.file_name}")
    print(f"Using evaluator model: {args.evaluator_model}")

    # Load prompt template
    prompt_template_path = Path(__file__).parent.parent / "prompts" / "extract_simulator_answer.txt"
    if not prompt_template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_template_path}")

    with open(prompt_template_path, 'r') as f:
        prompt_template = f.read()

    # Load annotations
    annotations = load_annotations(args.annotation_id)

    # Check for existing evaluations to avoid re-processing
    output_dir = Path(__file__).parent.parent / "evaluation_outputs" / "extracted_answer"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{args.file_name}.json"
    existing_evaluations = {}
    if output_file.exists():
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
            if "answers" in existing_data:
                existing_evaluations = existing_data["answers"]

    # Load simulation data
    simulation_data = load_simulation_data(args.file_name, args.annotation_id)
    terminated_data = load_terminated_conversations(args.file_name, args.annotation_id) if args.terminate_help else {}

    # Prepare batch prompts
    batch_prompts = []
    keys = []
    skipped = 0
    invalid = 0

    annotation_lookup = {ann["problem_id"]: ann for ann in annotations}

    for model_name in tqdm(simulation_data, desc="Processing models"):
        for problem_id in simulation_data[model_name]:
            problem_id_str = str(problem_id)

            # Get the math problem from annotations
            problem_id_int = int(problem_id) if isinstance(problem_id, str) else problem_id
            if problem_id_int not in annotation_lookup:
                print(f"Warning: No annotation found for problem {problem_id}")
                continue

            annotation = annotation_lookup[problem_id_int]
            math_problem = annotation["question"]

            for user_key, conversation_dict in simulation_data[model_name][problem_id_str].items():
                # Skip if already evaluated
                if (existing_evaluations and model_name in existing_evaluations and
                    problem_id_str in existing_evaluations[model_name] and
                    user_key in existing_evaluations[model_name][problem_id_str]):
                    skipped += 1
                    continue

                # Verify problem matches
                if math_problem != conversation_dict.get("problem", ""):
                    print(f"Warning: Problem mismatch for {problem_id}")
                    continue

                # Get termination turn if available
                terminate_turn = -1
                if args.terminate_help and terminated_data:
                    try:
                        terminate_turn = terminated_data[model_name][problem_id_str][user_key]["ending_turn_number"]
                    except (KeyError, TypeError):
                        pass

                # Build conversation text
                conversation = conversation_dict.get("assistant_messages", [])
                conversation_text = ""
                turn_num = 1

                for turn in conversation:
                    # Skip if past termination point
                    if args.terminate_help and terminate_turn > 0 and turn_num > terminate_turn:
                        break

                    if turn["role"] == "system":
                        continue

                    # Check for empty content
                    if not turn.get("content"):
                        print(f"Warning: Empty content in conversation for {model_name}/{problem_id}/{user_key}")
                        invalid += 1
                        break

                    if turn["role"] == "user":
                        # Use first query content for the first turn if available
                        if turn_num == 1 and "first_query_content" in conversation_dict:
                            query = conversation_dict["first_query_content"]
                        else:
                            query = turn["content"]
                        conversation_text += f"- Student at Turn {turn_num}: {query}\n"
                    else:  # assistant
                        conversation_text += f"- AI Tutor at Turn {turn_num}: {turn['content']}\n"
                        turn_num += 1

                if not conversation_text:
                    continue

                conversation_text = conversation_text.strip()

                # Create evaluation prompt
                prompt = prompt_template.format(
                    problem=math_problem,
                    conversation=conversation_text
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

    print(f"\nGenerated {len(batch_prompts)} prompts ({skipped} skipped as already evaluated, {invalid} invalid)")

    if len(batch_prompts) == 0:
        print("No new prompts to process. Exiting.")
        return

    # Save batch prompts in JSONL format for OpenAI batch API
    batch_dir = Path(__file__).parent.parent / "batch_prompts" / "extracted_answer"
    batch_dir.mkdir(parents=True, exist_ok=True)

    batch_file = batch_dir / f"{args.file_name}.jsonl"

    # Ensure parent directory exists (handles subdirectories in file_name like "gpt-5-mini/...")
    batch_file.parent.mkdir(parents=True, exist_ok=True)

    with open(batch_file, 'w') as f:
        for prompt in batch_prompts:
            f.write(json.dumps(prompt) + '\n')

    print(f"Batch prompts saved to: {batch_file}")

    # Save keys for later mapping
    keys_file = batch_dir / f"{args.file_name}_keys.json"
    keys_file.parent.mkdir(parents=True, exist_ok=True)
    with open(keys_file, 'w') as f:
        json.dump({"keys": [list(k) for k in keys]}, f, indent=2)

    print(f"Keys saved to: {keys_file}")

    return batch_file

if __name__ == "__main__":
    main()
