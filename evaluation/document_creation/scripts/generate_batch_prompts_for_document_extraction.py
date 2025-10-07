#!/usr/bin/env python3
"""
Generate batch prompts for document extraction in the document creation task.
This script prepares prompts to extract the final document from conversations.

The extracted documents are used for:
1. Document quality rating evaluation
2. Performance metrics computation
"""

import os
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm

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

def load_simulation_data(
    file_name: str,
    annotation_id: str
) -> Dict:
    """Load simulation output data from SimulatorArena."""
    simulation_path = Path(__file__).parent.parent.parent.parent / "simulation" / "output" / annotation_id / f"{file_name}.json"
    
    if not simulation_path.exists():
        raise FileNotFoundError(f"Simulation file not found: {simulation_path}")
    
    with open(simulation_path, 'r') as f:
        return json.load(f)

def load_terminated_conversations(
    file_name: str,
    annotation_id: str
) -> Dict:
    """Load terminated conversation data."""
    terminated_path = Path(__file__).parent.parent.parent.parent / "simulation" / "terminated_conversations" / annotation_id / f"{file_name}.json"
    
    if not terminated_path.exists():
        print(f"Warning: Terminated conversations file not found: {terminated_path}")
        print("Please run termination detection first. This is required for accurate document extraction.")
        return {}
    
    with open(terminated_path, 'r') as f:
        return json.load(f)

def load_existing_extractions(
    file_name: str,
    annotation_id: str
) -> Dict:
    """Load existing document extractions to avoid re-processing."""
    extraction_path = Path(__file__).parent.parent / "evaluation_outputs" / "extracted_document" / f"{file_name}.json"
    
    if not extraction_path.exists():
        return {}
    
    with open(extraction_path, 'r') as f:
        data = json.load(f)
        # Handle both direct format and wrapped format
        if "documents" in data:
            return data["documents"]
        return data

def prepare_conversation_text(
    conversation: List[Dict],
    terminate_turn: int = -1,
    terminate_help: bool = True
) -> Tuple[str, bool]:
    """
    Prepare conversation text for document extraction.
    
    Returns:
        Tuple of (conversation_text, contains_empty)
    """
    conversation_text = ""
    contains_empty = False
    turn_num = 1
    
    for turn in conversation:
        # Skip system messages
        if turn.get("role") == "system":
            continue
        
        # Stop at termination turn if specified
        if terminate_help and terminate_turn > 0 and turn_num > terminate_turn:
            break
        
        # Check for empty content
        content = turn.get("content", "")
        if not content:
            contains_empty = True
            break
        
        # Format the conversation
        if turn.get("role") == "user":
            conversation_text += f"- User Message at Turn {turn_num}: {content}\n"
        elif turn.get("role") == "assistant":
            conversation_text += f"- AI Writing Assistant Response at Turn {turn_num}: {content}\n"
            turn_num += 1
    
    return conversation_text.strip(), contains_empty

def main():
    parser = argparse.ArgumentParser(
        description="Generate batch prompts for document extraction from conversations."
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
        default="document_creation_annotations",
        help="Annotation dataset ID (default: document_creation_annotations)"
    )
    parser.add_argument(
        "--terminate_help",
        nargs='?',
        const=True,
        default=True,
        type=lambda x: str(x).lower() in ('true', '1', 'yes'),
        help="Use terminated conversation endpoints (default: True, HIGHLY RECOMMENDED)"
    )
    parser.add_argument(
        "--evaluator_model",
        type=str,
        default="gpt-5-mini",
        help="Model to use for document extraction (default: gpt-5-mini)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=5000,
        help="Maximum tokens for extracted document (default: 5000)"
    )
    
    args = parser.parse_args()
    
    print(f"Generating document extraction prompts for: {args.file_name}")
    print(f"Using evaluator model: {args.evaluator_model}")
    
    # Load prompt template
    prompt_template_path = Path(__file__).parent.parent / "prompts" / "extract_document.txt"
    if not prompt_template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_template_path}")
    
    with open(prompt_template_path, 'r') as f:
        prompt_template = f.read()
    
    # Load data
    simulation_data = load_simulation_data(args.file_name, args.annotation_id)
    terminated_data = load_terminated_conversations(args.file_name, args.annotation_id)
    
    if not terminated_data and args.terminate_help:
        print("ERROR: Terminated conversations are required for accurate document extraction.")
        print("Please run termination detection first:")
        print(f"  cd ../../simulation")
        print(f"  python terminate_conversation_document_creation.py --annotation_id {args.annotation_id} --simulation_path output/{args.annotation_id}/{args.file_name}.json")
        sys.exit(1)
    
    # Load existing extractions to avoid re-processing
    existing_extractions = load_existing_extractions(args.file_name, args.annotation_id)
    
    # Document type mapping (for consistency)
    document_type_dict = {
        "blog post": "Blog Post",
        "email": "Email/Letter",
        "creative writing": "Creative Writing",
    }
    
    # Prepare batch prompts
    batch_prompts = []
    keys = []
    skipped = 0
    empty_conversations = 0
    
    # Process simulation data
    # Data structure: data[model][document_type][intent][worker_id] = conversation_dict
    for model_name in tqdm(simulation_data, desc="Processing models"):
        for doc_type in simulation_data[model_name]:
            for intent in simulation_data[model_name][doc_type]:
                for worker_id, conversation_data in simulation_data[model_name][doc_type][intent].items():
                    # Create unique key for this conversation
                    key = f"{model_name}_{worker_id}_{doc_type}_{intent}"
                    
                    # Check if already extracted
                    if key in existing_extractions:
                        skipped += 1
                        continue
                    
                    # Get termination turn
                    terminate_turn = -1
                    if args.terminate_help and terminated_data:
                        try:
                            terminate_turn = terminated_data[model_name][doc_type][intent][worker_id]["ending_turn_number"]
                        except (KeyError, TypeError):
                            print(f"Warning: No termination turn found for {model_name}/{doc_type}/{intent}/{worker_id}")
                            # Skip if we require termination but don't have it
                            if args.terminate_help:
                                print(f"  Skipping due to missing termination data")
                                continue
                    
                    # Get conversation
                    conversation = conversation_data.get("assistant_messages", [])
                    
                    # Prepare conversation text
                    conversation_text, contains_empty = prepare_conversation_text(
                        conversation=conversation,
                        terminate_turn=terminate_turn,
                        terminate_help=args.terminate_help
                    )
                    
                    if contains_empty:
                        print(f"Warning: Empty content in conversation for {key}")
                        empty_conversations += 1
                        continue
                    
                    if not conversation_text:
                        print(f"Warning: No conversation text for {key}")
                        empty_conversations += 1
                        continue
                    
                    # Format prompt
                    prompt = prompt_template.format(conversation=conversation_text)
                    
                    # Create batch prompt in OpenAI format
                    batch_prompts.append({
                        "custom_id": key,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": args.evaluator_model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 1.0,
                            "max_completion_tokens": args.max_tokens,
                            "response_format": {"type": "json_object"}  # Ensure JSON output
                        }
                    })
                    keys.append(key)
    
    print(f"\nPrepared {len(batch_prompts)} document extraction prompts")
    print(f"  - Skipped {skipped} already extracted")
    print(f"  - Skipped {empty_conversations} empty conversations")
    
    if len(batch_prompts) == 0:
        print("\nNo new documents to extract. All conversations have been processed.")
        return None
    
    # Save batch prompts in JSONL format for OpenAI batch API
    batch_dir = Path(__file__).parent.parent / "batch_prompts" / "extracted_document"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    batch_file = batch_dir / f"{args.file_name}.jsonl"

    # Ensure parent directory exists (handles subdirectories in file_name like "gpt-5-mini/...")
    batch_file.parent.mkdir(parents=True, exist_ok=True)

    with open(batch_file, 'w') as f:
        for prompt in batch_prompts:
            f.write(json.dumps(prompt) + '\n')
    
    print(f"\nBatch prompts saved to: {batch_file}")
    
    # Save keys for later mapping
    keys_file = batch_dir / f"{args.file_name}_keys.json"
    keys_file.parent.mkdir(parents=True, exist_ok=True)
    with open(keys_file, 'w') as f:
        json.dump({
            "keys": keys,
            "total": len(keys),
            "model": args.evaluator_model,
            "terminate_help": args.terminate_help
        }, f, indent=2)
    
    print(f"Keys saved to: {keys_file}")
    
    # Save metadata for tracking
    metadata_file = batch_dir / f"{args.file_name}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            "file_name": args.file_name,
            "annotation_id": args.annotation_id,
            "evaluator_model": args.evaluator_model,
            "terminate_help": args.terminate_help,
            "total_prompts": len(batch_prompts),
            "skipped_existing": skipped,
            "empty_conversations": empty_conversations,
            "batch_file": str(batch_file.name),
            "keys_file": str(keys_file.name)
        }, f, indent=2)
    
    print(f"Metadata saved to: {metadata_file}")
    
    return batch_file

if __name__ == "__main__":
    main()