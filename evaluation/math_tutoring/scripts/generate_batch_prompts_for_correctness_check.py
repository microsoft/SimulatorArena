#!/usr/bin/env python3
"""
Evaluate the correctness of extracted student answers in math tutoring conversations.
Based on math_tutoring/user_simulation/extracted_simulator_answers/check_correctness_batch_prompt.py
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate math problem correctness using GPT-4o."
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="",
        help="File name (without extension) to process"
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="",
        help="File path to the data file. If provided, file_name is ignored"
    )
    parser.add_argument(
        "--annotation_id",
        type=str,
        default="good_annotations",
        help="Annotation ID (default: good_annotations)"
    )
    parser.add_argument(
        "--terminate_help",
        type=str2bool,
        default=True,
        help="Whether to use terminate_help version (default: True)"
    )
    parser.add_argument(
        "--extracted_answers_dir",
        type=str,
        default="",
        help="Directory containing extracted answers (default: evaluation_outputs/extracted_answer or extracted_simulator_answers)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="batch_prompts/check_correctness",
        help="Output directory for batch prompts"
    )
    
    args = parser.parse_args()
    
    # Determine file identifier
    file_identifier = args.file_name if args.file_name else args.file_path
    
    # Determine the extracted answers directory
    if not args.extracted_answers_dir:
        # Try new evaluation_outputs structure first
        evaluation_outputs_dir = Path(__file__).parent.parent / "evaluation_outputs" / "extracted_answer"
        if evaluation_outputs_dir.exists():
            args.extracted_answers_dir = str(evaluation_outputs_dir)
            print(f"Using evaluation_outputs directory: {args.extracted_answers_dir}")
        else:
            # Fall back to original location
            args.extracted_answers_dir = "extracted_simulator_answers"
            print(f"Using legacy directory: {args.extracted_answers_dir}")
    
    # Load extracted answers data
    if args.terminate_help:
        data_path = f"{args.extracted_answers_dir}/{args.annotation_id}_terminate_help/{file_identifier}.json"
    else:
        data_path = f"{args.extracted_answers_dir}/{args.annotation_id}/{file_identifier}.json"
    
    # Check if data file exists - try multiple locations
    if not os.path.exists(data_path):
        # Try simpler path in evaluation_outputs
        alt_paths = [
            Path(args.extracted_answers_dir) / f"{file_identifier}.json",
            Path(__file__).parent.parent / "evaluation_outputs" / "extracted_answer" / f"{file_identifier}.json"
        ]
        
        # If full path provided
        if args.file_path and os.path.exists(args.file_path):
            data_path = args.file_path
        else:
            # Try alternative paths
            found = False
            for alt_path in alt_paths:
                if alt_path.exists():
                    data_path = str(alt_path)
                    found = True
                    break
            
            if not found:
                # Try relative to script location for legacy support
                script_dir = Path(__file__).parent.parent.parent.parent
                legacy_path = script_dir / "math_tutoring" / "user_simulation" / "extracted_simulator_answers" / args.annotation_id
                if args.terminate_help:
                    legacy_path = legacy_path.parent / f"{args.annotation_id}_terminate_help"
                legacy_path = legacy_path / f"{file_identifier}.json"
                
                if legacy_path.exists():
                    data_path = str(legacy_path)
                else:
                    print(f"\nError: Cannot find extracted answers file!")
                    print(f"\nSearched in the following locations:")
                    print(f"  1. {data_path}")
                    for i, alt_path in enumerate(alt_paths, 2):
                        print(f"  {i}. {alt_path}")
                    print(f"  {len(alt_paths)+2}. {legacy_path}")
                    print(f"\nPlease ensure you have run answer extraction before running correctness check.")
                    print(f"Run: python generate_batch_prompts_for_answer_extraction.py --simulation_file <file> --annotation_file <file>")
                    print(f"Then: python process_batch_evaluation.py --batch_file <answer_extraction_batch_file> --wait")
                    sys.exit(1)
    
    print(f"Loading extracted answers from: {data_path}")
    with open(data_path, "r") as f:
        data = json.load(f)
    
    # Validate that this is an extracted answers file
    if not data or not isinstance(data, dict):
        print(f"Error: The file {data_path} does not appear to contain extracted answers.")
        sys.exit(1)
    
    # Check if this is an answers file (should have 'answers' key for new format or direct model keys for old format)
    if "answers" in data:
        # New format from evaluation_outputs
        data = data["answers"]
        print("Detected new format extracted answers file.")
    
    # Load annotations
    # In the original script, annotations are loaded from ../data relative to extracted_simulator_answers
    annotations_path = f"data/{args.annotation_id}.json"
    
    if not os.path.exists(annotations_path):
        # Try relative to script location
        script_dir = Path(__file__).parent.parent.parent.parent
        annotations_path = script_dir / "math_tutoring" / "user_simulation" / "data" / f"{args.annotation_id}.json"
        
        if not annotations_path.exists():
            print(f"Error: Cannot find annotations file at {annotations_path}")
            sys.exit(1)
    
    print(f"Loading annotations from: {annotations_path}")
    with open(annotations_path, "r") as f:
        annotations = json.load(f)
    
    # Load prompt template
    prompt_template_path = Path(__file__).parent.parent / 'prompts' / 'check_correctness.txt'
    
    if not prompt_template_path.exists():
        print(f"Error: Prompt template not found at {prompt_template_path}")
        sys.exit(1)
    
    with open(prompt_template_path, 'r') as f:
        prompt_template = f.read()
    
    # Build full contexts for the API call
    full_contexts = []
    keys = []
    no_clear_answer_num = 0
    already_evaluated = 0
    total_processed = 0
    no_extracted_answer_key = 0
    
    for model_name in data:
        if not isinstance(data[model_name], dict):
            continue
            
        for problem_id in data[model_name]:
            problem_id_str = str(problem_id)
            
            for turker_key in data[model_name][problem_id_str]:
                total_processed += 1
                
                # Skip if correctness has already been evaluated
                if "correctness" in data[model_name][problem_id_str][turker_key]:
                    already_evaluated += 1
                    continue
                
                # Find the matching annotation for this problem
                math_problem = None
                correct_answer = None
                
                for annotation in annotations:
                    if annotation["problem_id"] == int(problem_id):
                        math_problem = annotation["question"]
                        # Try different keys for the correct answer
                        correct_answer = (annotation.get("problem_1_gold_final_answer") or 
                                        annotation.get("answer") or 
                                        annotation.get("ground_truth_answer"))
                        break
                
                if not math_problem:
                    print(f"Warning: No math problem found for problem_id {problem_id}")
                    continue
                
                if not correct_answer:
                    print(f"Warning: No correct answer found for problem_id {problem_id}")
                    continue
                
                # Get the extracted student answer
                try:
                    simulator_answer = data[model_name][problem_id_str][turker_key].get("extracted_answer", "")
                except Exception as e:
                    print(f"Error extracting answer for {model_name} {problem_id} {turker_key}: {e}")
                    continue
                
                # Check if this entry has an extracted answer
                if "extracted_answer" not in data[model_name][problem_id_str][turker_key]:
                    no_extracted_answer_key += 1
                    continue
                
                # Skip if no clear answer was extracted
                if not simulator_answer or "No clear final answer" in simulator_answer:
                    no_clear_answer_num += 1
                    continue
                
                # Create evaluation prompt
                prompt = prompt_template.format(
                    question=math_problem,
                    correct_answer=correct_answer,
                    student_answer=simulator_answer
                )
                
                full_contexts.append([{"role": "user", "content": prompt}])
                keys.append((model_name, problem_id_str, turker_key))
    
    print(f"\nStatistics:")
    print(f"  Total conversations: {total_processed}")
    print(f"  Already evaluated: {already_evaluated}")
    print(f"  Missing extracted_answer key: {no_extracted_answer_key}")
    print(f"  No clear answer: {no_clear_answer_num}")
    print(f"  To be evaluated: {len(full_contexts)}")
    
    if no_extracted_answer_key > 0:
        print(f"\nWarning: {no_extracted_answer_key} conversations do not have extracted answers.")
        print("Please ensure answer extraction has been run on all conversations before checking correctness.")
    
    if len(full_contexts) == 0:
        print("\nNo new conversations to evaluate.")
        if no_extracted_answer_key > 0:
            print("\nError: Cannot check correctness without extracted answers.")
            print("Please run answer extraction first:")
            print("  1. python generate_batch_prompts_for_answer_extraction.py --simulation_file <file> --annotation_file <file>")
            print("  2. python process_batch_evaluation.py --batch_file <answer_extraction_batch_file> --wait")
            sys.exit(1)
        else:
            print("All conversations have already been evaluated for correctness.")
        return
    
    # Prepare batch data
    results_dict = {
        "keys": keys,
        "contexts": full_contexts,
        "metadata": {
            "task": "math_tutoring",
            "evaluation_type": "correctness",
            "annotation_id": args.annotation_id,
            "terminate_help": args.terminate_help,
            "extracted_answers_file": data_path,
            "annotations_file": str(annotations_path)
        }
    }
    
    # Save batch prompts
    if args.terminate_help:
        batch_saving_path = f"{args.output_dir}/{args.annotation_id}_terminate_help/{file_identifier}.json"
    else:
        batch_saving_path = f"{args.output_dir}/{args.annotation_id}/{file_identifier}.json"
    
    batch_saving_dir = os.path.dirname(batch_saving_path)
    
    # Create the directory if it doesn't exist
    os.makedirs(batch_saving_dir, exist_ok=True)
    
    with open(batch_saving_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    
    print(f"\nBatch prompts saved to: {batch_saving_path}")
    print(f"\nTo process this batch, run:")
    print(f"  python process_batch_evaluation.py --batch_file {batch_saving_path}")
    
    # Also save the path information for easy reference
    info_path = Path(batch_saving_path).parent / f"{Path(file_identifier).stem}_info.txt"
    with open(info_path, 'w') as f:
        f.write(f"Data source: {data_path}\n")
        f.write(f"Annotations: {annotations_path}\n")
        f.write(f"Prompt template: {prompt_template_path}\n")
        f.write(f"Total to evaluate: {len(full_contexts)}\n")
        f.write(f"No clear answer: {no_clear_answer_num}\n")
        f.write(f"Already evaluated: {already_evaluated}\n")

if __name__ == "__main__":
    main()