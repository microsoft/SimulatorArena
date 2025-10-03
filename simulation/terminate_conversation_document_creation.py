#!/usr/bin/env python3
"""
Script to generate and save terminated conversations for document creation simulations.
Usage: python terminate_conversation_document_creation.py --annotation_id ANNOTATION_ID --simulation_path SIMULATION_PATH
"""

import os
import re
import json
import argparse
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from utils import generate_from_azure_openai_chat_completion, merge_nested_dicts

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate and save terminated document creation conversations.')
    parser.add_argument('--annotation_id', type=str, required=True,
                        help='Annotation ID (e.g., good_annotations_50_benchmarking)')
    parser.add_argument('--simulation_path', type=str, required=True,
                        help='Path to simulation file (e.g., zero-shot-cot.json)')
    return parser.parse_args()

def get_ending_turn_number(text):
    """Extract the ending turn number from the generated text."""
    pattern = r'"Ending Turn Number"\s*:\s*(\d+)'
    match = re.search(pattern, text)
    if match:
        return int(match.group(1))
    return None

async def main():
    """Main function to process and generate terminated document creation conversations."""
    # Load environment variables
    dotenv_path = os.path.expanduser('~/.env')
    load_dotenv(dotenv_path)
    
    # Parse command-line arguments
    args = parse_arguments()
    annotation_id = args.annotation_id
    simulation_path = args.simulation_path
    
    # Read the termination prompt template
    prompt_path = Path(__file__).parent / 'prompts' / 'terminate_conversation_document_creation.txt'
    with open(prompt_path, 'r') as f:
        prompt_template = f.read()
    
    # Create base directory for terminated conversations if it doesn't exist
    output_base = Path(__file__).parent / 'terminated_conversations'
    output_base.mkdir(exist_ok=True)
    
    # Extract directory and filename from simulation_path
    simulation_dir = os.path.dirname(simulation_path)
    simulation_filename = os.path.basename(simulation_path)
    
    # Create the output directory structure
    output_dir = output_base / annotation_id
    if simulation_dir:
        output_dir = output_dir / simulation_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Path to the simulated conversations
    simulated_conversations_path = Path(__file__).parent / 'output' / annotation_id / simulation_path
    
    # Load the simulated conversations
    with open(simulated_conversations_path, "r") as f:
        simulated_conversations = json.load(f)
    
    # Load the annotations from SimulatorArena data
    annotations_path = Path(__file__).parent.parent / 'data' / 'document_creation_annotations.json'
    with open(annotations_path, "r") as f:
        annotations = json.load(f)
    
    # Define the save path for terminated conversations
    saved_path = output_dir / simulation_filename
    
    # Check if saved file exists and load existing conversations
    existing_conversations_dict = {}
    if saved_path.exists():
        with open(saved_path, "r") as f:
            existing_conversations_dict = json.load(f)
    
    print(f"Output will be saved to: {saved_path}")
    
    # Prepare data for generation
    saved_keys = []
    full_contexts = []
    turn_num_list = []
    
    for model in simulated_conversations:
        for document_type in simulated_conversations[model]:
            for intent in simulated_conversations[model][document_type]:
                for workerId in simulated_conversations[model][document_type][intent]:
                    
                    # Skip existing conversations if they exist in the saved file
                    if existing_conversations_dict:
                        if (model in existing_conversations_dict and 
                            document_type in existing_conversations_dict[model] and 
                            intent in existing_conversations_dict[model][document_type] and 
                            workerId in existing_conversations_dict[model][document_type][intent]):
                            continue
                    
                    conversation = simulated_conversations[model][document_type][intent][workerId]["assistant_messages"]
                    contains_empty = False
                    user_queries_text = ""
                    turn_num = 1
                    
                    for turn in conversation:
                        if turn["role"] == "system":
                            continue
                        if not turn["content"]:
                            contains_empty = True
                            break
                        if turn["role"] == "user":
                            query = turn["content"]
                            user_queries_text += f"- Turn {turn_num}: {query}\n"
                            turn_num += 1
    
                    turn_num = turn_num - 1
    
                    if contains_empty:
                        continue
                            
                    user_queries_text = user_queries_text.strip()
                    prompt = prompt_template.format(
                        document_type=document_type,
                        intent=intent,
                        user_messages=user_queries_text,
                    )
                    
                    saved_keys.append((model, workerId, document_type, intent))
                    full_contexts.append([{
                        "role": "user",
                        "content": prompt
                    }])
                    turn_num_list.append(turn_num)
    
    assert len(saved_keys) == len(full_contexts) == len(turn_num_list)
    print(f"Processing {len(saved_keys)} conversations...")
    
    if len(saved_keys) == 0:
        print("No new conversations to process. Exiting.")
        return
    
    # Generate responses
    responses = await generate_from_azure_openai_chat_completion(
        azure_resource_name="dl-openai-3",
        full_contexts=full_contexts,
        model_name="gpt-4o",
        temperature=0.7,
        max_tokens=4000,
        top_p=1.0,
        n=1,
        requests_per_minute=200,
        json_mode=False
    )
    
    # Process responses
    results_dict = {}
    for i, response in enumerate(responses):
        try:
            output = response.choices[0].message.content
        except Exception as e:
            print(f"Failed to extract output from response {i}: {e}")
            continue

        if not output:
            print(f"Empty response for conversation {i}")
            continue
            
        number = get_ending_turn_number(output)
        if not number:
            print(f"Failed to extract ending turn number from response {i}")
            print(output[:300] + "..." if len(output) > 300 else output)
            print("#" * 30)
            continue
            
        model, workerId, document_type, intent = saved_keys[i]
    
        if model not in results_dict:
            results_dict[model] = {}
        if document_type not in results_dict[model]:
            results_dict[model][document_type] = {}
        if intent not in results_dict[model][document_type]:
            results_dict[model][document_type][intent] = {}
    
        results_dict[model][document_type][intent][workerId] = {
            "output": output,
            "ending_turn_number": number,
            "original_turn_number": turn_num_list[i]
        }
    
    # Merge with existing results if any
    if existing_conversations_dict:
        results_dict = merge_nested_dicts(results_dict, existing_conversations_dict)
    
    # Save results
    with open(saved_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    
    print(f"Saved results to {saved_path}")

if __name__ == "__main__":
    asyncio.run(main())