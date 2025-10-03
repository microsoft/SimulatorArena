#!/usr/bin/env python3
"""
Extract final answers from math tutoring conversations.
Based on math_tutoring/user_simulation/extract_simulator_answer_batch_prompt.py
"""

import os
import json
import argparse
import re
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

def extract_student_answer(text):
    """
    Extracts the student's answer from the provided text using regex.
    Expects the answer to appear after "## Extracted Student's Answer:".
    """
    match = re.search(r'## Extracted Student\'s Answer:\s*(.*)', text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        return answer if answer else "Error"
    return "Error"

def merge_nested_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Merges two nested dictionaries with unique ending keys.
    """
    result = dict1.copy()
    
    def recursive_merge(current_dict, other_dict):
        for key, value in other_dict.items():
            if key not in current_dict:
                current_dict[key] = value
            elif isinstance(value, dict) and isinstance(current_dict[key], dict):
                recursive_merge(current_dict[key], value)
            else:
                print(f"Conflict at key: {key}. Keeping existing value.")
                continue
    
    recursive_merge(result, dict2)
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Extract simulator answers for math tutoring evaluation."
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="",
        help="The file name (without extension) in the output folder"
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="",
        help="The file path to the data file. If provided, file_name is ignored"
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
        help="Whether to terminate at help point (default: True)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="batch_prompts/extracted_simulator_answers",
        help="Output directory for batch prompts"
    )
    
    args = parser.parse_args()
    
    # Determine file identifier
    file_identifier = args.file_name if args.file_name else args.file_path
    
    # Load main data file
    data_path = f'output/{args.annotation_id}/{file_identifier}.json'
    
    # Check if data file exists
    if not os.path.exists(data_path):
        # Try alternative paths
        if args.file_path:
            data_path = args.file_path
        else:
            # Check if it's a full path
            if os.path.exists(file_identifier + '.json'):
                data_path = file_identifier + '.json'
            elif os.path.exists(file_identifier):
                data_path = file_identifier
            else:
                print(f"Error: Cannot find data file at {data_path}")
                sys.exit(1)
    
    print(f"Loading data from: {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Load annotations
    annotations_path = f"data/{args.annotation_id}.json"
    
    # Check if annotations file exists in standard location
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
    
    # Determine saving folder based on terminate_help flag
    if not args.terminate_help:
        saving_folder = f'extracted_simulator_answers/{args.annotation_id}'
    else:
        saving_folder = f'extracted_simulator_answers/{args.annotation_id}_terminate_help'
    
    if args.file_path:
        saving_folder = os.path.join(saving_folder, os.path.dirname(args.file_path))
    
    # Create output directory
    os.makedirs(saving_folder, exist_ok=True)
    saving_path = f"{saving_folder}/{os.path.basename(file_identifier)}.json"
    
    # Load prompt template
    prompt_template_path = Path(__file__).parent.parent / 'prompts' / 'extract_simulator_answer.txt'
    
    if not prompt_template_path.exists():
        print(f"Error: Prompt template not found at {prompt_template_path}")
        sys.exit(1)
    
    with open(prompt_template_path, 'r') as f:
        prompt_template = f.read()
    
    # Load terminated conversation turn numbers
    terminated_results_path = f"terminated_conversations/{args.annotation_id}/{file_identifier}.json"
    
    # Check if terminated results exist
    terminated_turn_num_results = {}
    if os.path.exists(terminated_results_path):
        with open(terminated_results_path, "r") as f:
            terminated_turn_num_results = json.load(f)
        print(f"Loaded termination data from: {terminated_results_path}")
    elif args.terminate_help:
        print(f"Warning: Termination data not found at {terminated_results_path}")
        print("Will process full conversations instead.")
        args.terminate_help = False
    
    # Load existing results if any
    existing_dict = {}
    if os.path.exists(saving_path):
        with open(saving_path, 'r') as f:
            existing_dict = json.load(f)
        print(f"Found existing results at: {saving_path}")
    
    full_contexts = []
    keys = []
    conversation_with_empty = 0
    skipped_existing = 0
    
    # Process each conversation in the data
    for model_name in data:
        if model_name not in data or not isinstance(data[model_name], dict):
            continue
            
        for problem_id in data[model_name]:
            problem_id_str = str(problem_id)
            
            for turker_key, conversation_dict in data[model_name][problem_id_str].items():
                # Skip if already processed
                if existing_dict:
                    if (model_name in existing_dict and 
                        problem_id_str in existing_dict[model_name] and 
                        turker_key in existing_dict[model_name][problem_id_str]):
                        skipped_existing += 1
                        continue
                
                # Get the math problem
                math_problem = conversation_dict.get("problem", "")
                
                # Get conversation
                conversation = conversation_dict.get("assistant_messages", [])
                conversation_text = ""
                contains_empty = False
                
                # Get termination turn number if available
                terminate_turn_num = float('inf')
                if args.terminate_help and terminated_turn_num_results:
                    try:
                        terminate_turn_num = terminated_turn_num_results[model_name][problem_id_str][turker_key]["ending_turn_number"]
                    except (KeyError, TypeError):
                        # If we can't find termination data for this conversation, skip it
                        pass
                
                turn_num = 1
                for turn in conversation:
                    # Stop at termination point if specified
                    if args.terminate_help and turn_num > terminate_turn_num:
                        break
                    
                    # Skip system messages
                    if turn.get("role") == "system":
                        continue
                    
                    # Check for empty content
                    if not turn.get("content"):
                        contains_empty = True
                        conversation_with_empty += 1
                        break
                    
                    # Format conversation
                    if turn["role"] == "user":
                        query = turn["content"]
                        # Use first_query_content for the first turn if available
                        if turn_num == 1 and "first_query_content" in conversation_dict:
                            query = conversation_dict["first_query_content"]
                        conversation_text += f"- Student at Turn {turn_num}: {query}\n"
                    else:  # assistant
                        conversation_text += f"- AI Tutor at Turn {turn_num}: {turn['content']}\n"
                        turn_num += 1
                
                conversation_text = conversation_text.strip()
                
                # Skip if conversation is empty or invalid
                if contains_empty or not conversation_text:
                    continue
                
                # Skip if conversation has odd number of messages (should end with student)
                # This check is from the original script
                if len(conversation) % 2 != 1:
                    conversation_with_empty += 1
                    continue
                
                # Create prompt
                prompt = prompt_template.format(
                    problem=math_problem, 
                    conversation=conversation_text
                )
                
                full_contexts.append([{"role": "user", "content": prompt}])
                keys.append((model_name, problem_id_str, turker_key))
    
    print(f"\nStatistics:")
    print(f"  Skipped (already processed): {skipped_existing}")
    print(f"  Conversations with empty/invalid turns: {conversation_with_empty}")
    print(f"  Total valid conversations to process: {len(full_contexts)}")
    
    if len(full_contexts) == 0:
        print("\nNo new conversations to process. Exiting.")
        return
    
    # Prepare batch data
    results_dict = {
        "keys": keys,
        "contexts": full_contexts,
        "metadata": {
            "task": "math_tutoring",
            "evaluation_type": "answer_extraction",
            "annotation_id": args.annotation_id,
            "terminate_help": args.terminate_help,
            "data_file": data_path,
            "annotations_file": str(annotations_path)
        }
    }
    
    # Save batch prompts
    batch_saving_path = os.path.join(args.output_dir, saving_folder, os.path.basename(file_identifier) + ".json")
    batch_saving_dir = os.path.dirname(batch_saving_path)
    
    # Create the directory if it doesn't exist
    os.makedirs(batch_saving_dir, exist_ok=True)
    
    with open(batch_saving_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    
    print(f"\nBatch prompts saved to: {batch_saving_path}")
    print(f"\nTo process this batch, run:")
    print(f"  python ../scripts/process_batch_evaluation.py --batch_file {batch_saving_path}")

if __name__ == "__main__":
    main()