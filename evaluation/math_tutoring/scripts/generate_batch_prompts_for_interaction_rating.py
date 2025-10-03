#!/usr/bin/env python3
"""
Interaction Rating Evaluation for Math Tutoring Task
Evaluates the quality of AI tutor interactions with students in math tutoring conversations.
This script prepares batch evaluation prompts for GPT-4 to rate interaction quality.
"""

import os
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from simulation.utils import merge_nested_dicts

def str2bool(v):
    """Converts a string to a boolean value for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def load_simulation_data(simulation_file: str) -> Dict:
    """Load simulation output data."""
    with open(simulation_file, 'r') as f:
        return json.load(f)

def load_annotations(annotation_file: str) -> List[Dict]:
    """Load annotation data."""
    with open(annotation_file, 'r') as f:
        return json.load(f)

def load_termination_data(termination_file: str) -> Dict:
    """Load conversation termination data if it exists."""
    if os.path.exists(termination_file):
        with open(termination_file, 'r') as f:
            return json.load(f)
    return {}

def prepare_evaluation_contexts(
    data: Dict,
    annotations: List[Dict],
    terminated_turns: Dict,
    prompt_template: str,
    terminate_at_help: bool = True,
    existing_evaluations: Dict = None
) -> Tuple[List[Dict], List[Tuple]]:
    """
    Prepare evaluation contexts for batch processing.
    
    Returns:
        Tuple of (contexts, keys) where contexts are the prompts and keys identify each evaluation
    """
    full_contexts = []
    keys = []
    existing_evaluations = existing_evaluations or {}
    
    # Create lookup for annotations by problem_id
    annotation_lookup = {ann["problem_id"]: ann for ann in annotations}
    
    for model_name in data:
        for problem_id in data[model_name]:
            problem_id_str = str(problem_id)
            
            # Get the math problem from annotations
            problem_id_int = int(problem_id) if isinstance(problem_id, str) else problem_id
            if problem_id_int not in annotation_lookup:
                print(f"Warning: No annotation found for problem {problem_id}")
                continue
                
            annotation = annotation_lookup[problem_id_int]
            math_problem = annotation["question"]
            
            for user_key, conversation_dict in data[model_name][problem_id_str].items():
                # Skip if already evaluated
                if (existing_evaluations and model_name in existing_evaluations and 
                    problem_id_str in existing_evaluations[model_name] and 
                    user_key in existing_evaluations[model_name][problem_id_str]):
                    continue
                
                # Verify problem matches
                if math_problem != conversation_dict.get("problem", ""):
                    print(f"Warning: Problem mismatch for {problem_id}")
                    continue
                
                # Get termination turn if available
                terminate_turn = None
                if terminated_turns and model_name in terminated_turns:
                    if problem_id_str in terminated_turns[model_name]:
                        if user_key in terminated_turns[model_name][problem_id_str]:
                            terminate_turn = terminated_turns[model_name][problem_id_str][user_key].get("ending_turn_number")
                
                # Build conversation text
                conversation = conversation_dict.get("assistant_messages", [])
                conversation_text = ""
                turn_num = 1
                
                for turn in conversation:
                    # Skip if past termination point
                    if terminate_at_help and terminate_turn and turn_num > terminate_turn:
                        break
                        
                    if turn["role"] == "system":
                        continue
                        
                    # Check for empty content
                    if not turn.get("content"):
                        print(f"Warning: Empty content in conversation for {model_name}/{problem_id}/{user_key}")
                        break
                        
                    if turn["role"] == "user":
                        # Use first query content for the first turn if available
                        if turn_num == 1 and "first_query_content" in conversation_dict:
                            query = conversation_dict["first_query_content"]
                        else:
                            query = turn["content"]
                        conversation_text += f"- Student Message at Turn {turn_num}: {query}\n"
                    else:  # assistant
                        conversation_text += f"- AI Tutor Response at Turn {turn_num}: {turn['content']}\n"
                        turn_num += 1
                
                if not conversation_text:
                    continue
                    
                conversation_text = conversation_text.strip()
                
                # Create evaluation prompt
                prompt = prompt_template.format(
                    problem=math_problem,
                    conversation=conversation_text
                )
                
                full_contexts.append([{
                    "role": "user",
                    "content": prompt
                }])
                keys.append((model_name, problem_id_str, user_key))
    
    return full_contexts, keys

def prepare_gold_human_contexts(
    annotations: List[Dict],
    prompt_template: str,
    existing_evaluations: Dict = None
) -> Tuple[List[Dict], List[Tuple]]:
    """
    Prepare evaluation contexts for gold human conversations.
    """
    full_contexts = []
    keys = []
    existing_evaluations = existing_evaluations or {}
    
    for annotation in annotations:
        model_name = annotation["model"]
        problem_id = str(annotation["problem_id"])
        annotation_key = f"{annotation['username']}_{annotation['workerId']}_{annotation['user_id']}"
        
        # Skip if already evaluated
        if (existing_evaluations and model_name in existing_evaluations and 
            problem_id in existing_evaluations[model_name] and 
            annotation_key in existing_evaluations[model_name][problem_id]):
            continue
        
        # Build conversation text from human annotations
        problem_turns = (annotation["problem_1_turns"] 
                        if annotation["problem_1_turns"] > 0 
                        else len(annotation["user_queries"]))
        
        ai_responses = annotation["ai_responses"][:problem_turns]
        user_queries = annotation["user_queries"][:problem_turns]
        
        conversation_text = ""
        for i, user_query in enumerate(user_queries):
            conversation_text += f"- Student Message at Turn {i + 1}: {user_query}\n"
            conversation_text += f"- AI Tutor Response at Turn {i + 1}: {ai_responses[i]}\n"
        
        conversation_text = conversation_text.strip()
        
        # Create evaluation prompt
        prompt = prompt_template.format(
            problem=annotation["question"],
            conversation=conversation_text
        )
        
        full_contexts.append([{
            "role": "user",
            "content": prompt
        }])
        keys.append((model_name, problem_id, annotation_key))
    
    return full_contexts, keys

def main():
    parser = argparse.ArgumentParser(
        description="Prepare batch evaluation prompts for math tutoring interaction rating."
    )
    parser.add_argument(
        "--simulation_file", 
        type=str, 
        required=True,
        help="Path to simulation output JSON file"
    )
    parser.add_argument(
        "--annotation_file", 
        type=str, 
        required=True,
        help="Path to annotation JSON file"
    )
    parser.add_argument(
        "--termination_file", 
        type=str, 
        default="",
        help="Path to termination data JSON file (optional)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="batch_prompts/interaction_rating",
        help="Output directory for batch evaluation prompts"
    )
    parser.add_argument(
        "--terminate_at_help", 
        type=str2bool, 
        default=True,
        help="Whether to terminate conversations at help point"
    )
    parser.add_argument(
        "--gold_human", 
        type=str2bool, 
        default=False,
        help="Evaluate gold human conversations instead of simulated ones"
    )
    parser.add_argument(
        "--evaluator_model", 
        type=str, 
        default="gpt-4o",
        help="Model to use for evaluation"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load prompt template
    prompt_file = f"../prompts/interaction_rating.txt"
    prompt_path = Path(__file__).parent / prompt_file
    
    if not prompt_path.exists():
        print(f"Error: Prompt template not found at {prompt_path}")
        sys.exit(1)
        
    with open(prompt_path, 'r') as f:
        prompt_template = f.read()
    
    # Load annotations
    annotations = load_annotations(args.annotation_file)
    
    # Prepare output file path
    if args.gold_human:
        output_filename = "gold_human"
    else:
        sim_basename = Path(args.simulation_file).stem
        output_filename = sim_basename
    
    if args.evaluator_model != "gpt-4o":
        output_filename += f"_{args.evaluator_model}"
    
    output_path = Path(args.output_dir) / f"{output_filename}.json"
    
    # Load existing evaluations if file exists
    existing_evaluations = {}
    if output_path.exists():
        with open(output_path, 'r') as f:
            existing_data = json.load(f)
            if "evaluations" in existing_data:
                existing_evaluations = existing_data["evaluations"]
    
    # Prepare evaluation contexts
    if args.gold_human:
        full_contexts, keys = prepare_gold_human_contexts(
            annotations, 
            prompt_template, 
            existing_evaluations
        )
    else:
        # Load simulation data
        data = load_simulation_data(args.simulation_file)
        
        # Load termination data if provided
        terminated_turns = {}
        if args.termination_file:
            terminated_turns = load_termination_data(args.termination_file)
        
        full_contexts, keys = prepare_evaluation_contexts(
            data,
            annotations,
            terminated_turns,
            prompt_template,
            args.terminate_at_help,
            existing_evaluations
        )
    
    print(f"Prepared {len(full_contexts)} evaluation prompts")
    
    # Save batch evaluation data
    batch_data = {
        "metadata": {
            "task": "math_tutoring",
            "evaluation_type": "interaction_rating",
            "evaluator_model": args.evaluator_model,
            "terminate_at_help": args.terminate_at_help,
            "gold_human": args.gold_human,
            "simulation_file": args.simulation_file if not args.gold_human else None,
            "annotation_file": args.annotation_file,
            "termination_file": args.termination_file if args.termination_file else None
        },
        "keys": keys,
        "contexts": full_contexts
    }
    
    with open(output_path, 'w') as f:
        json.dump(batch_data, f, indent=2)
    
    print(f"Batch evaluation prompts saved to: {output_path}")
    
    # Print summary statistics
    model_counts = {}
    for model_name, _, _ in keys:
        model_counts[model_name] = model_counts.get(model_name, 0) + 1
    
    print("\nEvaluation summary by model:")
    for model, count in sorted(model_counts.items()):
        print(f"  {model}: {count} conversations")

if __name__ == "__main__":
    main()