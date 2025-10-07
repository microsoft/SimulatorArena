#!/usr/bin/env python3
"""
Generate batch prompts for rating evaluation in document creation task.
Supports both document quality rating and interaction quality rating.

This script prepares evaluation prompts for batch processing via OpenAI's API.
"""

import os
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
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
        print("Using full conversations. Consider running termination detection first.")
        return {}
    
    with open(terminated_path, 'r') as f:
        return json.load(f)

def load_annotations(annotation_id: str) -> List[Dict]:
    """Load annotation data from SimulatorArena data folder."""
    annotations_path = Path(__file__).parent.parent.parent.parent / "data" / "document_creation_annotations.json"
    
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
    
    with open(annotations_path, 'r') as f:
        return json.load(f)

def load_user_preferences(annotation_id: str = "document_creation_annotations") -> Dict:
    """Load user preferences for document creation."""
    preferences_path = Path(__file__).parent.parent.parent.parent / "data" / "user_simulator_profiles" / "document_creation" / "preferences.json"
    
    if not preferences_path.exists():
        print(f"Warning: Preferences file not found: {preferences_path}")
        return {}
    
    with open(preferences_path, 'r') as f:
        return json.load(f)

def load_user_background(annotation_id: str = "document_creation_annotations") -> Dict:
    """Load user background information."""
    background_path = Path(__file__).parent.parent.parent.parent / "data" / "document_creation_user_simulator_background.json"
    
    if not background_path.exists():
        print(f"Warning: Background file not found: {background_path}")
        return {}
    
    with open(background_path, 'r') as f:
        return json.load(f)

def extract_final_document(conversation: List[Dict]) -> str:
    """
    Extract the final document from a conversation.
    The final document is typically in the last assistant message.
    """
    # Look for the last assistant message that contains substantial content
    for message in reversed(conversation):
        if message.get("role") == "assistant" and message.get("content"):
            content = message["content"].strip()
            # Check if this looks like a document (more than just a short response)
            if len(content) > 100:  # Assuming documents are at least 100 chars
                return content
    
    return ""

def prepare_document_rating_prompt(
    final_document: str,
    document_type: str,
    intent: str,
    preferences: List[Dict],
    prompt_template: str
) -> str:
    """Prepare a prompt for document quality rating."""
    
    # Format preferences
    document_preferences_text = ""
    if preferences:
        for feature in preferences:
            document_preferences_text += f'- {feature.get("Preference Name", "")}: {feature.get("Preference Question Answer", "")}\n'
    document_preferences_text = document_preferences_text.strip()
    
    # Fill in the template
    prompt = prompt_template.format(
        document_type=document_type,
        intent=intent,
        document_preferences=document_preferences_text,
        final_document=final_document
    )
    
    return prompt

def prepare_interaction_rating_prompt(
    conversation: List[Dict],
    terminate_turn: int,
    prompt_template: str
) -> str:
    """Prepare a prompt for interaction quality rating."""
    
    conversation_text = ""
    turn_num = 1
    
    for message in conversation:
        # Skip system messages
        if message.get("role") == "system":
            continue
        
        # Stop at termination turn if specified
        if terminate_turn > 0 and turn_num > terminate_turn:
            break
        
        content = message.get("content", "").strip()
        if not content:
            continue
        
        if message.get("role") == "user":
            conversation_text += f"- User Message at Turn {turn_num}: {content}\n"
        elif message.get("role") == "assistant":
            conversation_text += f"- AI Writing Assistant Response at Turn {turn_num}: {content}\n"
            turn_num += 1
    
    conversation_text = conversation_text.strip()
    
    # Fill in the template
    prompt = prompt_template.format(conversation=conversation_text)
    
    return prompt

def main():
    parser = argparse.ArgumentParser(
        description="Generate batch prompts for document creation evaluation (rating)."
    )
    
    # Required arguments
    parser.add_argument(
        "--file_name",
        type=str,
        required=True,
        help="Name of the simulation output file (without .json extension)"
    )
    parser.add_argument(
        "--aspect",
        type=str,
        required=True,
        choices=["document", "interaction"],
        help="Evaluation aspect: 'document' for final document quality, 'interaction' for conversation quality"
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
        help="Use terminated conversation endpoints if available (default: True)"
    )
    parser.add_argument(
        "--evaluator_model",
        type=str,
        default="gpt-5-mini",
        help="Model to use for evaluation (default: gpt-5-mini)"
    )
    parser.add_argument(
        "--gold_human",
        action="store_true",
        help="Evaluate human conversations instead of simulated ones"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=5000,
        help="Maximum tokens for rating response (default: 5000)"
    )

    args = parser.parse_args()
    
    print(f"Generating {args.aspect} rating prompts for: {args.file_name}")
    print(f"Using evaluator model: {args.evaluator_model}")
    
    # Load prompt template
    prompt_template_path = Path(__file__).parent.parent / "prompts" / f"{args.aspect}_rating.txt"
    if not prompt_template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_template_path}")
    
    with open(prompt_template_path, 'r') as f:
        prompt_template = f.read()
    
    # Load data
    if not args.gold_human:
        simulation_data = load_simulation_data(args.file_name, args.annotation_id)
        terminated_data = load_terminated_conversations(args.file_name, args.annotation_id) if args.terminate_help else {}
    
    annotations = load_annotations(args.annotation_id)
    preferences_dict = load_user_preferences()
    background_dict = load_user_background()
    
    # Check for existing evaluations to avoid re-processing
    output_dir = Path(__file__).parent.parent / "evaluation_outputs" / f"{args.aspect}_rating"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{args.file_name}.json"
    existing_evaluations = {}
    if output_file.exists():
        with open(output_file, 'r') as f:
            existing_evaluations = json.load(f)
            if "evaluations" in existing_evaluations:
                existing_evaluations = existing_evaluations["evaluations"]
    
    # Document type mapping
    document_type_dict = {
        "blog post": "Blog Post",
        "email": "Email/Letter",
        "creative writing": "Creative Writing",
    }
    reverse_document_type_dict = {v: k for k, v in document_type_dict.items()}
    
    # Prepare batch prompts
    batch_prompts = []
    keys = []
    skipped = 0
    
    if args.gold_human:
        # Process human annotations
        for annotation in tqdm(annotations, desc="Processing human annotations"):
            model_name = annotation["model"]
            doc_type = document_type_dict.get(annotation["document_type"], annotation["document_type"])
            intent = annotation["intent"]
            worker_id = annotation["workerId"]
            
            # Check if already evaluated
            if (model_name in existing_evaluations and
                doc_type in existing_evaluations[model_name] and
                intent in existing_evaluations[model_name][doc_type] and
                worker_id in existing_evaluations[model_name][doc_type][intent]):
                skipped += 1
                continue
            
            # Prepare conversation or document
            if args.aspect == "interaction":
                conversation = []
                for i, (user_q, ai_resp) in enumerate(zip(annotation.get("user_queries", []), 
                                                           annotation.get("ai_responses", []))):
                    conversation.append({"role": "user", "content": user_q})
                    conversation.append({"role": "assistant", "content": ai_resp})
                
                prompt = prepare_interaction_rating_prompt(
                    conversation=conversation,
                    terminate_turn=-1,  # Use full conversation for human
                    prompt_template=prompt_template
                )
            
            elif args.aspect == "document":
                # Get final document from annotation
                document_history = annotation.get("document_history", [])
                final_document = document_history[-1] if document_history else ""
                
                if not final_document:
                    print(f"Warning: Empty document for {model_name}/{doc_type}/{intent}/{worker_id}")
                    continue
                
                # Get preferences
                pref_key = f'{model_name}_{worker_id}_{annotation["document_type"]}_{intent}'
                preferences = preferences_dict.get(pref_key, [])
                
                prompt = prepare_document_rating_prompt(
                    final_document=final_document,
                    document_type=doc_type,
                    intent=intent,
                    preferences=preferences,
                    prompt_template=prompt_template
                )
            
            batch_prompts.append({
                "custom_id": f"{model_name}|{doc_type}|{intent}|{worker_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.evaluator_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 1.0,
                    "max_completion_tokens": args.max_tokens
                }
            })
            keys.append((model_name, doc_type, intent, worker_id))
    
    else:
        # Process simulation data
        for model_name in tqdm(simulation_data, desc="Processing models"):
            for doc_type in simulation_data[model_name]:
                for intent in simulation_data[model_name][doc_type]:
                    for worker_id, conversation_data in simulation_data[model_name][doc_type][intent].items():
                        # Check if already evaluated
                        if (model_name in existing_evaluations and
                            doc_type in existing_evaluations[model_name] and
                            intent in existing_evaluations[model_name][doc_type] and
                            worker_id in existing_evaluations[model_name][doc_type][intent]):
                            skipped += 1
                            continue
                        
                        # Get termination turn if available
                        terminate_turn = -1
                        if args.terminate_help and terminated_data:
                            try:
                                terminate_turn = terminated_data[model_name][doc_type][intent][worker_id]["ending_turn_number"]
                            except (KeyError, TypeError):
                                pass
                        
                        # Get conversation
                        conversation = conversation_data.get("assistant_messages", [])
                        
                        if args.aspect == "interaction":
                            prompt = prepare_interaction_rating_prompt(
                                conversation=conversation,
                                terminate_turn=terminate_turn,
                                prompt_template=prompt_template
                            )
                        
                        elif args.aspect == "document":
                            # Extract final document
                            if terminate_turn > 0:
                                # Use conversation up to termination point
                                truncated_conv = []
                                turn_count = 0
                                for msg in conversation:
                                    if msg.get("role") != "system":
                                        if msg.get("role") == "assistant":
                                            turn_count += 1
                                        if turn_count > terminate_turn:
                                            break
                                    truncated_conv.append(msg)
                                final_document = extract_final_document(truncated_conv)
                            else:
                                final_document = extract_final_document(conversation)
                            
                            if not final_document:
                                print(f"Warning: Could not extract document for {model_name}/{doc_type}/{intent}/{worker_id}")
                                continue
                            
                            # Get preferences
                            # Try to find preferences with flexible model matching
                            pref_key = f'{model_name}_{worker_id}_{reverse_document_type_dict.get(doc_type, doc_type)}_{intent}'
                            preferences = preferences_dict.get(pref_key, [])
                            
                            # If not found, try with alternative model names
                            if not preferences:
                                allowed_models = ["gpt-4o-mini", "gpt-4o", "llama-3-1-70b", "llama-3-1-8b", 
                                                "phi-3-small", "gpt-4-turbo", "phi-3-medium", 
                                                "mistral-large-2407", "claude-3-5-sonnet-20240620"]
                                for alt_model in allowed_models:
                                    alt_key = f'{alt_model}_{worker_id}_{reverse_document_type_dict.get(doc_type, doc_type)}_{intent}'
                                    if alt_key in preferences_dict:
                                        preferences = preferences_dict[alt_key]
                                        break
                            
                            prompt = prepare_document_rating_prompt(
                                final_document=final_document,
                                document_type=doc_type,
                                intent=intent,
                                preferences=preferences,
                                prompt_template=prompt_template
                            )
                        
                        batch_prompts.append({
                            "custom_id": f"{model_name}|{doc_type}|{intent}|{worker_id}",
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": args.evaluator_model,
                                "messages": [{"role": "user", "content": prompt}],
                                "temperature": 1.0,
                                "max_completion_tokens": args.max_tokens
                            }
                        })
                        keys.append((model_name, doc_type, intent, worker_id))
    
    print(f"\nGenerated {len(batch_prompts)} prompts ({skipped} skipped as already evaluated)")
    
    # Save batch prompts in JSONL format for OpenAI batch API
    batch_dir = Path(__file__).parent.parent / "batch_prompts" / f"{args.aspect}_rating"
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