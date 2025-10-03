#!/usr/bin/env python3
import os
import json
import asyncio
import argparse
import traceback
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

# Import your utilities (adjust paths if necessary)
from utils import (
    simulate_conversation_in_batch_document_creation,
    simulate_conversation_with_user_profile_in_batch_document_creation,
    construct_filename,
    merge_nested_dicts
)

# Add this constant near the top of the file (or keep here).
MODEL_TIMEOUT_MIN = 130
MODEL_TIMEOUT_SEC = MODEL_TIMEOUT_MIN * 60   # 150 minutes

# ---------------------------
# Utility functions
# ---------------------------
def round_down_to_nearest_5(n):
    """Rounds down to the nearest 5, with a minimum of 1."""
    return max(1, (n // 5) * 5)

def round_up_to_nearest_5(n):
    """Rounds up to the nearest 5 (minimum of 1)."""
    if n <= 0:
        return 1
    return ((n + 4) // 5) * 5

def count_words(text):
    return len([word for word in text.split() if word])


# ---------------------------
# Asynchronous functions for simulation
# ---------------------------
async def process_model(model, document_dict, user_model, 
                        prompt_initial_query_template, prompt_template, show_progress,
                        user_profile=False, refinement=False, length_control=False,
                        refinement_version="v1"):
    # Debug prints for progress monitoring
    print(f"Model {model} has {len(document_dict['document_type'])} documents, "
          f"running with user_profile={user_profile}, refinement={refinement}, "
          f"length_control={length_control}, with user_model={user_model}")

    document_types = document_dict["document_type"]
    intents = document_dict["intent"]
    backgrounds = document_dict["background"]
    workerIds = document_dict["workerId"]

    if user_profile:
        simulated_conversations = await simulate_conversation_with_user_profile_in_batch_document_creation(
            document_types, intents, backgrounds, document_dict["user_profile"],
            user_model, model, prompt_initial_query_template, prompt_template,
            user_temperature=0.7, assistant_temperature=0, show_progress=show_progress,
            length_control_bool=length_control,
            length_control_list=document_dict["length_control"],
            refinement=refinement,
            user_query_style_profiles=document_dict["user_query_style_profile"],
            refinement_version=refinement_version,
        )
    else:
        simulated_conversations = await simulate_conversation_in_batch_document_creation(
            document_types, intents, backgrounds, user_model, model,
            prompt_initial_query_template, prompt_template,
            user_temperature=0.7, assistant_temperature=0, show_progress=show_progress,
            length_control_bool=length_control,
            length_control_list=document_dict["length_control"],
            refinement=refinement,
            user_query_style_profiles=document_dict["user_query_style_profile"],
            refinement_version=refinement_version,
        )

    assert len(simulated_conversations) == len(document_types), "Unexpected mismatch in conversation lengths"

    model_data = {}
    for i, simulated_conversation in enumerate(simulated_conversations):
        doc_type = document_types[i]
        intent = intents[i]
        # Remove keys not needed
        simulated_conversation.pop('first_query', None)
        simulated_conversation.pop('conversation_history', None)

        if doc_type not in model_data:
            model_data[doc_type] = {}
        if intent not in model_data[doc_type]:
            model_data[doc_type][intent] = {}
        model_data[doc_type][intent][workerIds[i]] = simulated_conversation

    return model, model_data

async def run_all_models(
    model_document_dict,
    user_model,
    prompt_initial_query_template,
    prompt_template,
    show_progress=True,
    user_profile=False,
    refinement=False,
    length_control=False,
    refinement_version="v1",
):
    tasks = []
    for model, document_dict in model_document_dict.items():

        async def timed_process(model_name=model, doc_dict=document_dict):
            """Run process_model with a per-model timeout."""
            try:
                return await asyncio.wait_for(
                    process_model(
                        model_name,
                        doc_dict,
                        user_model,
                        prompt_initial_query_template,
                        prompt_template,
                        show_progress,
                        user_profile=user_profile,
                        refinement=refinement,
                        length_control=length_control,
                        refinement_version=refinement_version,
                    ),
                    timeout=MODEL_TIMEOUT_SEC,
                )
            except asyncio.TimeoutError:
                print(f"[TIMEOUT] {model_name} exceeded {MODEL_TIMEOUT_MIN} min and was skipped.")
                return None, None  # maintain tuple shape

        tasks.append(asyncio.create_task(timed_process()))

    model_simulated_conversations = {}
    for completed_task in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        try:
            model_name, model_data = await completed_task
            if model_name and model_data:            # ignore timed-out runs
                model_simulated_conversations[model_name] = model_data
        except Exception:
            traceback.print_exc()

    print(f"Successful models: {list(model_simulated_conversations.keys())}")
    return model_simulated_conversations



# ---------------------------
# Main asynchronous entry point
# ---------------------------
async def main(args):
    # (Optionally) change working directory to the script's directory
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Load environment variables (if needed)
    load_dotenv(os.path.expanduser('~/.env'))

    # Retrieve command-line arguments
    version                = args.version
    user_profile_version   = args.user_profile_version
    annotation_id          = args.annotation_id
    length_control         = args.length_control
    length_control_setting = args.length_control_setting
    refinement             = args.refinement
    refinement_message_style = args.refinement_message_style
    user_model            = args.user_model
    refinement_version = args.refinement_version
    benchmarking           = args.benchmarking
    allowed_models_str     = args.allowed_models

    # Determine whether to use user profiles based on version and provided flag
    if "user-profile" in version and user_profile_version != "":
        use_user_profile = True
    else:
        use_user_profile = False

    # Load background data
    with open(f"../data/document_creation_user_background.json", "r") as f:
        background_dict = json.load(f)

    # Load user profile and preference data (used if user_profile_version or refinement_message_style is set)
    with open(f"../data/user_simulator_profiles/document_creation/interaction_style.json", "r") as f:
        extracted_interaction_style_user_profile_dict = json.load(f)

    with open(f"../data/user_simulator_profiles/document_creation/writing_style.json", "r") as f:
        extracted_writing_style_user_profile_dict = json.load(f)

    with open(f"../data/user_simulator_profiles/document_creation/preferences.json", "r") as f:
        extracted_preferences_dict = json.load(f)

    # Load prompt templates
    with open(f'prompts/document_creation/{version}.txt', 'r') as f:
        prompt_template = f.read()
    with open(f'prompts/document_creation/{version}-initial-query.txt', 'r') as f:
        prompt_initial_query_template = f.read()

    # Construct output file name using provided settings
    file_name = construct_filename(
        version,
        user_profile_version=user_profile_version,
        length_control=length_control,
        length_control_setting=length_control_setting,
        refinement=refinement,
        refinement_message_style=refinement_message_style,
        refinement_version=refinement_version,
    )
    # Add benchmarking suffix if benchmarking mode is enabled
    if benchmarking:
        file_name += "_benchmarking"
    print(f"Output filename: {file_name}")

    # Load annotations
    with open(f"../data/{annotation_id}.json", "r") as f:
        annotations = json.load(f)
    print(f"Total annotations loaded: {len(annotations)}")

    # Filter out annotations that have already been simulated
    existing_output = {}
    document_type_dict = {
        "blog post": "Blog Post",
        "email": "Email/Letter",
        "creative writing": "Creative Writing",
    }
    if user_model == "gpt-4o":
        output_dir = f"output/{annotation_id}"
    else:
        output_dir = f"output/{annotation_id}/{user_model}"

    if os.path.exists(f"{output_dir}/{file_name}.json"):
        with open(f"{output_dir}/{file_name}.json", "r") as f:
            existing_output = json.load(f)
        new_annotations = []
        for annotation in annotations:
            try:
                # Try to access the result; if not found, include the annotation.
                _ = existing_output[annotation["model"]][document_type_dict[annotation["document_type"]]][annotation["intent"]][annotation["workerId"]]
            except KeyError:
                new_annotations.append(annotation)
        annotations = new_annotations
        print("After filtering already simulated annotations, there are", len(annotations), "annotations left")

    # Build background_list from annotations
    background_list = []
    for annotation in annotations:
        key = f'{annotation["model"]}_{annotation["workerId"]}_{annotation["document_type"]}_{annotation["intent"]}'
        background = background_dict[key]
        background_text = ""
        for bullet_point in background:
            background_text += f"- {bullet_point['question']}\n"
            background_text += f"    {bullet_point['answer']}\n"
        background_list.append(background_text.strip())

    # Build length control texts if enabled
    length_control_list = []
    if length_control:
        for i, annotation in tqdm(enumerate(annotations), desc="Computing length control"):
            user_queries = annotation["user_queries"]
            query_length_list = [count_words(query) for query in user_queries]
            if length_control_setting == "range":
                rounded_min = int(round_down_to_nearest_5(min(query_length_list)))
                rounded_max = int(round_up_to_nearest_5(max(query_length_list)))
                length_control_text = f"between {rounded_min} and {rounded_max} words"
            elif length_control_setting == "average":
                average_user_query_length = sum(query_length_list) / len(query_length_list)
                length_control_text = f"around {int(round_up_to_nearest_5(average_user_query_length))} words"
            else:
                raise ValueError(f"Length control setting {length_control_setting} not supported")
            length_control_list.append(length_control_text)

    # Build user profile texts if needed (for both profile and refinement)
    user_profile_list = []
    user_query_style_profiles = []
    if user_profile_version != "" or refinement_message_style != "":
        for i, annotation in tqdm(enumerate(annotations), desc="Computing user profiles"):
            key = f'{annotation["model"]}_{annotation["workerId"]}_{annotation["document_type"]}_{annotation["intent"]}'
            writing_style = extracted_writing_style_user_profile_dict[key]
            interaction_style = extracted_interaction_style_user_profile_dict[key]
            preferences = extracted_preferences_dict[key]
            user_queries = annotation["user_queries"]
            query_length_list = [count_words(query) for query in user_queries]
            if length_control_setting == "range":
                rounded_min = int(round_down_to_nearest_5(min(query_length_list)))
                rounded_max = int(round_up_to_nearest_5(max(query_length_list)))
                length_control_text = f"between {rounded_min} and {rounded_max} words"
            elif length_control_setting == "average":
                average_user_query_length = sum(query_length_list) / len(query_length_list)
                length_control_text = f"around {int(round_up_to_nearest_5(average_user_query_length))} words"
            else:
                raise ValueError(f"Length control setting {length_control_setting} not supported")

            user_profile_text = ""
            user_query_style_text = ""
            if "preference" in user_profile_version:
                user_profile_text += "\n## Document Preferences\n"
                for feature in preferences:
                    user_profile_text += f"- {feature['Preference Name']}: {feature['Preference Question Answer']}\n"
            if "preference" in refinement_message_style:
                user_query_style_text = user_profile_text  
            if "writing" in user_profile_version:
                user_profile_text += "\n## Writing Style\n"
                for feature in writing_style:
                    user_profile_text += f"- {feature['Feature Name']}: {feature['Feature Question Answer']}\n"
            if "writing" in refinement_message_style:
                user_query_style_text += "\n## Writing Style\n"
                for feature in writing_style:
                    user_query_style_text += f"- {feature['Feature Name']}: {feature['Feature Question Answer']}\n"
            if "interaction" in user_profile_version:
                user_profile_text += "\n## Interaction Style\n"
                user_profile_text += f"- Length of Message: The user's message is always {length_control_text}.\n"
                for feature in interaction_style:
                    user_profile_text += f"- {feature['Feature Name']}: {feature['Feature Question Answer']}\n"
            if "interaction" in refinement_message_style:
                user_query_style_text += "\n## Interaction Style\n"
                user_query_style_text += f"- Length of Message: The user's message is always {length_control_text}.\n"
                for feature in interaction_style:
                    user_query_style_text += f"- {feature['Feature Name']}: {feature['Feature Question Answer']}\n"


            user_profile_list.append(user_profile_text.strip())
            user_query_style_profiles.append(user_query_style_text.strip())

    

    # Build a dictionary mapping models to their corresponding simulation inputs
    if benchmarking:
        # In benchmarking mode, use user-specified models
        if not allowed_models_str:
            raise ValueError("--allowed_models must be specified when --benchmarking is enabled")
        allowed_models = [model.strip() for model in allowed_models_str.split(",")]
        print(f"Benchmarking mode enabled with models: {allowed_models}")
    else:
        # Default allowed models when not benchmarking
        allowed_models = ["gpt-4o-mini", "gpt-4o",
                          "mistral-large-2407", "claude-3-5-sonnet-20240620", 
                          "llama-3-1-70b", "llama-3-1-8b", "phi-3-small", "gpt-4-turbo", "phi-3-medium"]

    model_document_dict = {}

    if benchmarking:
        # In benchmarking mode, run all specified models for each annotation
        for i, annotation in enumerate(annotations):
            for model_name in allowed_models:
                # Skip if this annotation already exists in the output for this model
                if existing_output:
                    try:
                        _ = existing_output[model_name][document_type_dict[annotation["document_type"]]][annotation["intent"]][annotation["workerId"]]
                        continue
                    except KeyError:
                        pass
                
                if model_name not in model_document_dict:
                    model_document_dict[model_name] = {
                        "document_type": [],
                        "intent": [],
                        "workerId": [],
                        "background": [],
                        "user_profile": [],
                        "user_initial_understanding_profile": [],
                        "user_query_style_profile": [],
                        "length_control": [],
                    }
                
                doc_type = document_type_dict[annotation["document_type"]]
                model_document_dict[model_name]["document_type"].append(doc_type)
                model_document_dict[model_name]["intent"].append(annotation["intent"])
                model_document_dict[model_name]["workerId"].append(annotation["workerId"])
                model_document_dict[model_name]["background"].append(background_list[i])

                if length_control:
                    model_document_dict[model_name]["length_control"].append(length_control_list[i])
                if use_user_profile:
                    model_document_dict[model_name]["user_profile"].append(user_profile_list[i])
                if refinement:
                    model_document_dict[model_name]["user_query_style_profile"].append(user_query_style_profiles[i])
    else:
        # Non-benchmarking mode: use model from annotation
        for i, annotation in enumerate(annotations):
            if annotation["model"] not in allowed_models:
                continue

            if annotation["model"] not in model_document_dict:
                model_document_dict[annotation["model"]] = {
                    "document_type": [],
                    "intent": [],
                    "workerId": [],
                    "background": [],
                    "user_profile": [],
                    "user_initial_understanding_profile": [],
                    "user_query_style_profile": [],
                    "length_control": [],
                }
            doc_type = document_type_dict[annotation["document_type"]]
            model_document_dict[annotation["model"]]["document_type"].append(doc_type)
            model_document_dict[annotation["model"]]["intent"].append(annotation["intent"])
            model_document_dict[annotation["model"]]["workerId"].append(annotation["workerId"])
            model_document_dict[annotation["model"]]["background"].append(background_list[i])

            if length_control:
                model_document_dict[annotation["model"]]["length_control"].append(length_control_list[i])
            if use_user_profile:
                model_document_dict[annotation["model"]]["user_profile"].append(user_profile_list[i])
            if refinement:
                model_document_dict[annotation["model"]]["user_query_style_profile"].append(user_query_style_profiles[i])


    # Run the simulation asynchronously across models
    model_simulated_conversations = await run_all_models(
        model_document_dict,
        user_model,
        prompt_initial_query_template,
        prompt_template,
        show_progress=False,
        user_profile=use_user_profile,
        refinement=refinement,
        length_control=length_control,
        refinement_version=refinement_version,
    )

    # Merge with any existing output
    if existing_output:
        final_output = merge_nested_dicts(existing_output, model_simulated_conversations)
    else:
        final_output = model_simulated_conversations

    # Ensure output directory exists and save results
    os.makedirs(output_dir, exist_ok=True)
    out_file = f"{output_dir}/{file_name}.json"
    with open(out_file, "w") as f:
        json.dump(final_output, f, indent=4)
    print(f"Simulation results saved to: {out_file}")


# ---------------------------
# Command-line interface
# ---------------------------
def cli_parser():
    parser = argparse.ArgumentParser(description="Async simulation runner with user profiles.")
    parser.add_argument("--version", type=str, required=True,
                        help="Version string (default: zero-shot-cot).")
    parser.add_argument("--user_profile_version", type=str, default="",
                        help="User profile version string (default: empty).")
    parser.add_argument("--annotation_id", type=str, default="document_creation_annotations",
                        help="Annotation ID (default: document_creation_annotations).")
    parser.add_argument("--length_control", action="store_true",
                        help="Flag for length control (default: False).")
    parser.add_argument("--length_control_setting", type=str, default="range",
                        help="Length control setting (default: range).")
    parser.add_argument("--refinement", action="store_true",
                        help="Flag for refinement (default: False).")
    parser.add_argument("--refinement_version", type=str, default="v1",
                        help="Refinement version (default: v1).")
    parser.add_argument("--refinement_message_style", type=str, default="",
                        help="Refinement message style (default: "").")
    parser.add_argument("--user_model", type=str, default="gpt-4o",
                        help="User model (default gpt-4o).")
    parser.add_argument("--benchmarking", action="store_true",
                        help="Enable benchmarking mode (default: False).")
    parser.add_argument("--allowed_models", type=str, default="",
                        help="Comma-separated list of models to benchmark (e.g., 'gpt-5,claude-sonnet-4-20250514').")
    return parser

if __name__ == "__main__":
    parser = cli_parser()
    args = parser.parse_args()
    asyncio.run(main(args))
