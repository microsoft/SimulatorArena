#!/usr/bin/env python3
import os
import json
import asyncio
import traceback
import argparse
import random

from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

# Import your simulation functions and helpers from utils
from utils import (
    simulate_conversation_in_batch_math_tutoring,
    simulate_conversation_with_user_profile_in_batch_math_tutoring,
    construct_filename,
    merge_nested_dicts,
)

# Put this near your imports or above the function if you haven’t already.
MODEL_TIMEOUT_MIN = 130               # minutes
MODEL_TIMEOUT_SEC = MODEL_TIMEOUT_MIN * 60

#########################
# Helper Functions
#########################
def round_down_to_nearest_5(n):
    """Rounds down to the nearest 5, with a minimum of 1."""
    return max(1, (n // 5) * 5)

def round_up_to_nearest_5(n):
    """Rounds up to the nearest 5, with a minimum of 1."""
    if n <= 0:
        return 1
    return ((n + 4) // 5) * 5

def count_words(text):
    return len([word for word in text.split() if word])

#########################
# CLI Argument Parser
#########################
def cli_parser():
    parser = argparse.ArgumentParser(description="Async math tutoring simulation runner.")
    parser.add_argument("--version", type=str, required=True,
                        help="Version string (default: "").")
    parser.add_argument("--user_profile_version", type=str, default="",
                        help="User profile version (default: "").")
    parser.add_argument("--length_control", action="store_true",
                        help="Enable length control (default: False).")
    parser.add_argument("--length_control_setting", type=str, default="range",
                        help="Length control setting (range or average, default: range).")
    parser.add_argument("--refinement", action="store_true",
                        help="Enable refinement (default: False).")
    parser.add_argument("--refinement_version", type=str, default="v1",
                        help="Refinement version (default: v1).")
    parser.add_argument("--refinement_message_style", type=str, default="",
                        help="Refinement message style (default: empty).")
    parser.add_argument("--annotation_id", type=str, default="math_tutoring_annotations",
                        help="Annotation ID (default: good_annotations).")
    parser.add_argument("--seed", type=int, default=2,
                        help="Random seed (default: 2).")
    parser.add_argument("--user_model", type=str, default="gpt-4o",
                        help="User model (default gpt-4o).")
    return parser

#########################
# Asynchronous Simulation Functions
#########################
async def process_model(
    model, 
    problem_dict, 
    user_model, 
    prompt_initial_query_template, 
    prompt_template, 
    show_progress,
    user_profile=False,
    refinement=False,
    length_control=False,
    refinement_version="v1",
):
    print(f"Model {model} has {len(problem_dict['problem'])} problems. with user model being {user_model}")
    print(f"    user_profile: {user_profile}, refinement: {refinement}, length_control: {length_control}")
    
    problems = problem_dict["problem"]
    problem_ids = problem_dict["problem_id"]
    saved_keys = problem_dict["saved_key"]

    if user_profile:
        simulated_conversations = await simulate_conversation_with_user_profile_in_batch_math_tutoring(
            problems, problem_dict["user_profile"], user_model, model,
            prompt_initial_query_template, prompt_template,
            user_temperature=0.7, assistant_temperature=0,
            show_progress=show_progress, refinement=refinement, refinement_version=refinement_version,
            user_query_style_profiles=problem_dict.get("user_query_style_profile", []),
            length_control_bool=length_control,
            length_control_list=problem_dict.get("length_control", []),
        )
    else:
        simulated_conversations = await simulate_conversation_in_batch_math_tutoring(
            problems, user_model, model, prompt_initial_query_template, prompt_template,
            user_temperature=0.7, assistant_temperature=0, show_progress=show_progress, 
            length_control_bool=length_control, length_control_list=problem_dict.get("length_control", []),
            refinement=refinement, refinement_version=refinement_version, user_query_style_profiles=problem_dict.get("user_query_style_profile", [])
        )

    assert len(simulated_conversations) == len(problems), "Mismatch in conversation lengths."

    # Collect results by problem_id and saved_key
    model_data = {}
    for i, simulated_conversation in enumerate(simulated_conversations):
        pid = problem_ids[i]
        skey = saved_keys[i]
        # Remove keys not needed in final output
        simulated_conversation.pop('first_query', None)
        simulated_conversation.pop('conversation_history', None)
        if pid not in model_data:
            model_data[pid] = {}
        model_data[pid][skey] = simulated_conversation

    return model, model_data

async def run_all_models(
    model_problem_dict,
    user_model,
    prompt_initial_query_template=None,
    prompt_template=None,
    show_progress=True,
    user_profile=False,
    sample_and_select=False,
    refinement=False,
    length_control=False,
    refinement_version="v1",
):
    tasks = []
    for model, problem_dict in model_problem_dict.items():

        # code.interact(local=dict(globals(), **locals()))

        async def timed_process(model_name=model, prob_dict=problem_dict):
            """Wrap process_model with a per-model timeout."""
            try:
                return await asyncio.wait_for(
                    process_model(
                        model_name, prob_dict, user_model,
                        prompt_initial_query_template, prompt_template,
                        show_progress,
                        user_profile=user_profile,
                        sample_and_select=sample_and_select,
                        refinement=refinement,
                        length_control=length_control,
                        refinement_version=refinement_version,
                    ),
                    timeout=MODEL_TIMEOUT_SEC,
                )
            except asyncio.TimeoutError:
                print(f"[TIMEOUT] {model_name} exceeded {MODEL_TIMEOUT_MIN} min and was skipped.")
                return None, None  # preserve tuple structure for downstream unpacking

        task = asyncio.create_task(timed_process())
        tasks.append(task)

    model_simulated_conversations = {}
    for completed in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        try:
            model, model_data = await completed
            if model and model_data:
                model_simulated_conversations[model] = model_data
        except Exception:
            traceback.print_exc()

    print(f"Successful models: {list(model_simulated_conversations.keys())}")
    return model_simulated_conversations

#########################
# Main Async Entry Point
#########################
async def main(args):
    # Load environment variables from ~/.env
    load_dotenv(os.path.expanduser('~/.env'))

    # Set random seed
    random.seed(args.seed)

    # Set flags and parameters from args (some flags can be derived from version if needed)
    version                = args.version
    user_profile_version   = args.user_profile_version
    length_control         = args.length_control
    length_control_setting = args.length_control_setting
    refinement             = args.refinement
    refinement_version      = args.refinement_version
    refinement_message_style = args.refinement_message_style
    annotation_id          = args.annotation_id
    user_model             = args.user_model

    if refinement:
        assert refinement_message_style, "Refinement message style must be provided if refinement is enabled."

    # Determine some flags based on version
    # For example, if the version string contains "user-profile", then set user_profile True.
    user_profile = ("user-profile" in version) and (user_profile_version != "")

    # Construct a filename for output
    file_name = construct_filename(
        version,
        user_profile_version=user_profile_version, length_control=length_control,
        length_control_setting=length_control_setting,
        refinement=refinement, refinement_message_style=refinement_message_style,
        refinement_version=refinement_version
    )
    print(f"Output filename: {file_name}")

    #########################
    # Load Prompts and User Profiles
    #########################
    prompt_template_path = f"prompts/math_tutoring/{version}.txt"
    with open(prompt_template_path, 'r') as f:
        prompt_template = f.read()
        prompt_initial_query_template_path = f"prompts/math_tutoring/{version}-initial-query.txt"
        with open(prompt_initial_query_template_path, 'r') as f:
            prompt_initial_query_template = f.read()

    # Load user profiles and related data if user_profile is enabled or for refinement
    writing_profile_path = f"../data/user_profiles/math_tutoring/writing_style.json"
    interaction_profile_path = f"../data/user_profiles/math_tutoring/interaction_style.json"
    knowledge_state_profile_path = f"../data/user_profiles/math_tutoring/knowledge_state.json"
    with open(writing_profile_path, 'r') as f:
        extracted_writing_style_user_profile_dict = json.load(f)
    with open(interaction_profile_path, 'r') as f:
        extracted_interaction_style_user_profile_dict = json.load(f)
    with open(knowledge_state_profile_path, 'r') as f:
        extracted_knowledge_state_user_profile_dict = json.load(f)
    
    # Load extracted concepts for problems
    with open("../data/math_tutoring_problems_extracted_concepts.json", 'r') as f:
        extracted_concepts = json.load(f)

    # Define some dictionaries for math tutoring
    math_expertise_levels = {
        "k-5": "Elementary School (Grades K-5)",
        "6-8": "Middle School (Grades 6-8)",
        "9-12": "High School (Grades 9-12)",
        "undergrad": "Undergraduate-level",
        "grad": "Graduate-level",
        "research": "Conduct studies or research"
    }
    why_ask_tutor_dict = {
        "L1": "I'm completely lost and don't understand the problem",
        "L2": "I understand the problem, but I don't know where to start",
        "L3": "I've written some steps, but now I'm stuck",
        "L4": "I can solve the problem, but I'm not confident in my solution",
        "Other reason": "Other reason"
    }

    #########################
    # Load Annotations and Turker Conversations
    #########################
    annotations = []
    annotation_path = f"../data/{annotation_id}.json"
    with open(annotation_path, "r") as f:
        annotations = json.load(f)
    print(f"Loaded {len(annotations)} annotations.")

    # Filter annotations: if user_profile, require 'why_ask_tutor' field.
    filtered_annotations = []
    for ann in annotations:
        if user_profile and ("why_ask_tutor" not in ann):
            continue
        filtered_annotations.append(ann)
    annotations = filtered_annotations
    print(f"After filtering data that has no why_ask_tuor, {len(annotations)} annotations remain.")

    # If output already exists, filter out annotations that have been simulated already.
    if user_model == "gpt-4o":
        output_dir = f"output/{annotation_id}"
    else:
        output_dir = f"output/{annotation_id}/{user_model}"
    out_file_path = f"{output_dir}/{file_name}.json"

    # code.interact(local=dict(globals(), **locals())) # for debugging
    existing_output = {}
    if os.path.exists(out_file_path):
        with open(out_file_path, "r") as f:
            existing_output = json.load(f)
        new_annotations = []
        for ann in annotations:
            try:
                # Try to access the simulated output; if missing, keep the annotation.
                _ = existing_output[ann["model"]][str(ann["problem_id"])][ann["workerId"]]
            except KeyError:
                new_annotations.append(ann)
        annotations = new_annotations
        print(f"After filtering already simulated annotations, {len(annotations)} remain.")

    #########################
    # Process Annotations: Build User Profiles, and Length Control Info
    #########################
    user_profile_list = []
    user_initial_understanding_profiles = []
    user_query_style_profiles = []
    length_control_list = []

    # Loop through annotations to construct additional info.
    for i, ann in tqdm(enumerate(annotations), desc="Processing annotations"):
        key = repr((ann["problem_id"], ann["user_id"], ann["model"]))
        # --- Length control ---
        user_queries = ann["user_queries"]
        problem_turns = ann["problem_1_turns"] if ann["problem_1_turns"] > 0 else len(user_queries)
        user_queries = user_queries[:problem_turns]
        query_length_list = [count_words(query) for query in user_queries]
        
        if length_control_setting == "range":
            rounded_min = int(round_down_to_nearest_5(min(query_length_list)))
            rounded_max = int(round_up_to_nearest_5(max(query_length_list)))
            length_text = f"between {rounded_min} and {rounded_max} words"
        elif length_control_setting == "average":
            avg_length = sum(query_length_list) / len(query_length_list)
            length_text = f"around {int(round_up_to_nearest_5(avg_length))} words"
        else:
            raise ValueError(f"Unsupported length_control_setting: {length_control_setting}")
        
        if length_control:
            length_control_list.append(length_text)
        # --- User Profile / Initial Understanding / Query Style ---
        profile_text = ""
        initial_understanding_text = ""
        query_style_text = ""
        if "knowledge_state" in user_profile_version:
            profile_text += "\n## Initial Knowledge State\n"
            profile_text += f"- Expertise Level: {math_expertise_levels.get(ann['math_expertise'], 'N/A')}\n"
            profile_text += f"- Problem Understanding: {why_ask_tutor_dict.get(ann['why_ask_tutor'], 'N/A')}\n"
            profile_text += "- Concepts Understanding:\n"
            concept_num = 1
            concepts = extracted_concepts[str(ann["problem_id"])]["extracted_concepts"]
            for concept in concepts:
                concept_name = concept["Concept Name"]
                concept_explanation = concept["Concept Explanation"]
                # Find status from knowledge state profile
                status = None
                for c in extracted_knowledge_state_user_profile_dict.get(key, {}).get("concepts", []):
                    if c["Concept Name"] == concept_name:
                        status = c["Status"]
                        break
                if not status or status == "Not introduced":
                    continue
                profile_text += f"  {concept_num}. {concept_name}\n"
                profile_text += f"     Description: {concept_explanation}\n"
                profile_text += f"     Status: {status}\n\n"
                concept_num += 1
            initial_understanding_text = profile_text
        if "knowledge_state" in refinement_message_style:
            query_style_text = profile_text
        if "knowledge_level_short" in user_profile_version:
            profile_text = "\n## Initial Knowledge State\n"
            profile_text += f"- Expertise Level: {math_expertise_levels.get(ann['math_expertise'], 'N/A')}\n"
            profile_text += f"- Problem Understanding: {why_ask_tutor_dict.get(ann['why_ask_tutor'], 'N/A')}\n"
        if "writing" in user_profile_version:
            profile_text += "\n## Writing Style\n"
            for feature in extracted_writing_style_user_profile_dict.get(key, []):
                profile_text += f"- {feature['Feature Name']}: {feature['Feature Question Answer']}\n"
        if "writing" in refinement_message_style:
            query_style_text += "\n## Writing Style\n"
            for feature in extracted_writing_style_user_profile_dict.get(key, []):
                query_style_text += f"- {feature['Feature Name']}: {feature['Feature Question Answer']}\n"
        if "interaction" in user_profile_version:
            profile_text += "\n## Interaction Style\n"
            profile_text += f"- Length of User Message: The user's query/response is always {length_text}.\n"
            for feature in extracted_interaction_style_user_profile_dict.get(key, []):
                profile_text += f"- {feature['Feature Name']}: {feature['Feature Question Answer']}\n"
        if "interaction" in refinement_message_style:
            query_style_text += "\n## Interaction Style\n"
            query_style_text += f"- Length of User Message: The user's query/response is always {length_text}.\n"
            for feature in extracted_interaction_style_user_profile_dict.get(key, []):
                query_style_text += f"- {feature['Feature Name']}: {feature['Feature Question Answer']}\n"
     
        # code.interact(local=dict(globals(), **locals())) # for debugging
        user_profile_list.append(profile_text.strip())
        user_initial_understanding_profiles.append(initial_understanding_text.strip())
        user_query_style_profiles.append(query_style_text.strip())

    # End loop through annotations

    # Ensure all profile-related lists are aligned with annotations
    if user_profile:
        assert len(user_profile_list) == len(annotations)
        assert len(user_initial_understanding_profiles) == len(annotations)
        assert len(user_query_style_profiles) == len(annotations)
    if length_control:
        assert len(length_control_list) == len(annotations)

    #########################
    # Build model-problem dictionary
    #########################

    allowed_models = ["gpt-4o-mini", "gpt-4o", "llama-3-1-70b", 
                      "llama-3-1-8b", "phi-3-small", "gpt-4-turbo", 
                      "mistral-large-latest", "claude-3-5-sonnet-20240620", "phi-3-medium"]


    model_problem_dict = {}

    model_count = {}

    for i, ann in enumerate(annotations):
        if ann["model"] not in allowed_models:
            continue

        if ann["model"] not in model_problem_dict:
            model_problem_dict[ann["model"]] = {
                "problem": [],
                "problem_id": [],
                "saved_key": [],
                "why_ask_tutor": [],
                "user_profile": [],
                "user_initial_understanding_profile": [],
                "user_query_style_profile": [],
                "length_control": [],
            }
        model_problem_dict[ann["model"]]["problem"].append(ann["question"])
        saved_key = ann["workerId"]
        model_problem_dict[ann["model"]]["saved_key"].append(saved_key)
        model_problem_dict[ann["model"]]["problem_id"].append(str(ann["problem_id"]))
        model_problem_dict[ann["model"]]["why_ask_tutor"].append(ann["why_ask_tutor"])
        if user_profile:
            model_problem_dict[ann["model"]]["user_profile"].append(user_profile_list[i])
        if refinement:
            model_problem_dict[ann["model"]]["user_query_style_profile"].append(user_query_style_profiles[i])
        if length_control:
            model_problem_dict[ann["model"]]["length_control"].append(length_control_list[i])


    #########################
    # Run Simulation
    #########################
    if user_profile:
        model_simulated_conversations = await run_all_models(
            model_problem_dict, user_model,
            prompt_initial_query_template, prompt_template,
            show_progress=False, user_profile=True,
            refinement=refinement, length_control=length_control,
            refinement_version=refinement_version
        )
    else:
        model_simulated_conversations = await run_all_models(
            model_problem_dict, user_model,
            prompt_initial_query_template, prompt_template,
            show_progress=False, length_control=length_control, refinement=refinement,
            refinement_version=refinement_version
        )

    #########################
    # Merge with existing output (if any) and save results
    #########################
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if existing_output:
        final_output = merge_nested_dicts(existing_output, model_simulated_conversations)
    else:
        final_output = model_simulated_conversations

    with open(f"{output_dir}/{file_name}.json", "w") as f:
        json.dump(final_output, f, indent=4)
    print(f"Simulation results saved to: {output_dir}/{file_name}.json")

if __name__ == "__main__":
    parser = cli_parser()
    args = parser.parse_args()
    asyncio.run(main(args))