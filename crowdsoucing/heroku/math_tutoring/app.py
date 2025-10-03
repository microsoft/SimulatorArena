import gradio as gr
import os

import time

import re

import uuid

import random

import pandas as pd
from openai import AzureOpenAI, OpenAI
import anthropic

from mistralai import Mistral

from huggingface_hub import HfApi

from logger_config import logger

import json

from utils import create, handle_latex_delimeter, persist

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential

from dotenv import load_dotenv
dotenv_path = os.path.expanduser('~/.env')
load_dotenv(dotenv_path)
os.environ['CURL_CA_BUNDLE'] = ''

openai_client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))

anthropic_client = anthropic.Anthropic()

mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

llama_3_1_70b_client = ChatCompletionsClient(endpoint=os.environ.get("AZURE_LLAMA_3_1_70B_INSTRUCT_ENDPOINT"),
                                             credential=AzureKeyCredential(os.environ.get("AZURE_LLAMA_3_1_70B_INSTRUCT_KEY")))

llama_3_1_8b_client = ChatCompletionsClient(endpoint=os.environ.get("AZURE_LLAMA_3_1_8B_INSTRUCT_ENDPOINT"),
                                            credential=AzureKeyCredential(os.environ.get("AZURE_LLAMA_3_1_8B_INSTRUCT_KEY")))

phi_3_medium_client = ChatCompletionsClient(endpoint=os.environ.get("AZURE_PHI3_MEDIUM_128K_INSTRUCT_ENDPOINT"),
                                            credential=AzureKeyCredential(os.environ.get("AZURE_PHI3_MEDIUM_128K_INSTRUCT_KEY")))

phi_3_small_client = ChatCompletionsClient(endpoint=os.environ.get("AZURE_PHI3_SMALL_128K_INSTRUCT_ENDPOINT"),
                                             credential=AzureKeyCredential(os.environ.get("AZURE_PHI3_SMALL_128K_INSTRUCT_KEY")))

DATA_PATH = "data/problem_bank_1000.csv"
math_data = pd.read_csv(DATA_PATH)

# Shuffle the data and reset the index
math_data = math_data.sample(frac=1).reset_index(drop=True)

# Add the index column
math_data['index'] = math_data.index

# Reorder columns to place 'index' as the first column
cols = ['index'] + [col for col in math_data if col != 'index']
math_data = math_data[cols]

# Create the concise dataframe with the extra 'index' column
math_data_concise = math_data[['index', 'problem_id', 'problem', 'level', 'type']]

# read cookies.json
with open("cookies.json", "r") as f:
    cookies = json.load(f)
    
with open("worker_user_id_dict.json", "r") as f:
    worker_user_id_dict = json.load(f)

FOLDER_PATH = os.getenv("ANNOTATIONS_DIR", "annotations")
if not os.path.exists(FOLDER_PATH):
    os.makedirs(FOLDER_PATH)

DATASET_REPO_URL = os.getenv("HF_REPO_ID", "")
HF_TOKEN = os.getenv("HF_TOKEN")

hf_api = HfApi(token=HF_TOKEN)

latex_delimeter_set = [
    {"left": "\\\\[", "right": "\\\\]", "display": True}, {"left": "\\\\(", "right": "\\\\)", "display": False},
    {"left": "\\[", "right": "\\]", "display": True}, {"left": "\\(", "right": "\\)", "display": False},
    {"left": "$$", "right": "$$", "display": True}, {"left": "$", "right": "$", "display": False},
    {"left": "\\begin{equation}", "right": "\\end{equation}", "display": True},
    {"left": "\\begin{align}", "right": "\\end{align}", "display": True},
    {"left": "\\begin{align*}", "right": "\\end{align*}", "display": True},
    {"left": "\\begin{alignat}", "right": "\\end{alignat}", "display": True},
    {"left": "\\begin{gather}", "right": "\\end{gather}", "display": True},
    {"left": "\\begin{CD}", "right": "\\end{CD}", "display": True},
]

def load_instance(request: gr.Request):
    query_params = dict(request.query_params)
    username = query_params.get("username", "anonymous")
    folder_path = os.path.join(FOLDER_PATH, username)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # for mturk user set user_id to their previous user_id in case they remove the cookies
    if query_params.get("workerId", "") in worker_user_id_dict:
        user_id = worker_user_id_dict[query_params.get("workerId", "")]
    else:
        if "user_id" in request.cookies and request.cookies["user_id"] != "null":
            user_id = request.cookies["user_id"]
        else:
            user_id = str(uuid.uuid4())

    model_name = query_params.get("model", "")
        
    start_time = time.time()

    state_dict = {
            "username": username,
            "user_id": user_id,
            "math_expertise": "",
            "dataset": "math",
            "datapath": DATA_PATH,
            "assignmentId": query_params.get("assignmentId", ""),
            "hitId": query_params.get("hitId", ""),
            "workerId": query_params.get("workerId", ""),
            "turkSubmitTo": query_params.get("turkSubmitTo", ""),
            "cheat": query_params.get("cheat", ""),
            "user_queries": [],
            "ai_responses": [],
            "turk_solution": "",
            "turk_final_answer": "",
            "turk_problem_2_solution": "",
            "turk_problem_2_final_answer": "",
            "strength_weakness": "",
            "why_stop": "",
            "overall_rating": 0,
            "moved_to_problem_2": False,
            "problem_1_turns": -1,
            "start_time": start_time,
            "model": model_name,
        }

    user_id_dict = {"user_id": user_id}
    if not state_dict["assignmentId"]:
        return state_dict, user_id_dict, \
                gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), \
                gr.update(visible=False) # for turker_bonus_instruction
    else:
        if state_dict["assignmentId"] == "ASSIGNMENT_ID_NOT_AVAILABLE":
            return state_dict, user_id_dict, \
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
                gr.update(visible=True) # for turker_bonus_instruction
        else:
            return state_dict, user_id_dict, \
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), \
                gr.update(visible=True) # for turker_bonus_instruction
                
def vote(state, data: gr.LikeData):
    if state["moved_to_problem_2"]:
        if "problem_2_turn_level_vote" not in state:
            state["problem_2_turn_level_vote"] = {}
        state["problem_2_turn_level_vote"][repr(data.index)] = [data.value, data.liked]
    else:
        if "turn_level_vote" not in state:
            state["turn_level_vote"] = {}
        state["turn_level_vote"][repr(data.index)] = [data.value, data.liked]
    return state
    
    
def echo(message, history, state):
    history_openai_format = [
        {"role": "system", "content": "You are a skilled math tutor. Your goal is to help students understand and solve problems independently. Provide guidance based on their questions or mistakes. Ask questions to encourage their thinking and let students do most of the work themselves. Never give out the solution directly to students."}
        ]

    for i, (human, assistant) in enumerate(history):
        if i == 0:
            user_message = "Here is the problem that you will tutor me on:\n" + state["question"].strip() + "\n\n"
            history_openai_format.append({"role": "user", "content": user_message + human})
        elif i == state["problem_1_turns"]:
            user_message = "Here is the second problem that has similar concepts as the first problem. You will tutor me on this problem now.\n\n" + state["similar_question"].strip() + "\n\n"
            history_openai_format.append({"role": "user", "content": user_message + human})
        else:
            history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content":assistant})

    if not history:
        user_message = "Here is the problem that you will tutor me on:\n" + state["question"].strip() + "\n\n"
        history_openai_format.append({"role": "user", "content": user_message + message.text})
    elif len(history) == state["problem_1_turns"]:
        user_message = "Here is the second problem that has similar concepts as the first problem. You will tutor me on this problem now.\n\n" + state["similar_question"].strip() + "\n\n"
        history_openai_format.append({"role": "user", "content": user_message + message.text})
    else:
        history_openai_format.append({"role": "user", "content": message.text})

    if state["model"] in ["gpt-4o", "gpt-35-turbo", "gpt-4-turbo", "gpt-4o-mini"]:

        model_name = state["model"]
        if model_name == "gpt-35-turbo":
            model_name = "gpt-3.5-turbo"
        if model_name == "gpt-4o":
            model_name = "gpt-4o-2024-05-13"
        try:
            response = openai_client.chat.completions.create(
                model = model_name,
                messages = history_openai_format,
                temperature = 0,
                max_tokens=2000,
                n=1,
                stream=True
            )
        except Exception as e:
            raise gr.Error(f"An error occurred: {e}")
        
        partial_message = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                partial_message = partial_message + chunk.choices[0].delta.content

                partial_message = handle_latex_delimeter(partial_message)
                yield partial_message
        
    elif state["model"] in ["mistral-large-latest"]:
        # With streaming
        try:
            stream_response = mistral_client.chat.stream(model="mistral-large-2407",
                                                          messages=history_openai_format,
                                                          temperature=0,
                                                          max_tokens=2000)
        except Exception as e:
            raise gr.Error(f"An error occurred: {e}")

        partial_message = ""
        for chunk in stream_response:
            if chunk.data.choices[0].delta.content is not None:
                partial_message = partial_message + chunk.data.choices[0].delta.content

                partial_message = handle_latex_delimeter(partial_message)
                yield partial_message

    elif state["model"] in ["claude-3-5-sonnet-20240620"]:
        partial_message = ""
        try:
            with anthropic_client.messages.stream(
                max_tokens=2000,
                messages=history_openai_format[1:], # anthropic does not support system message in messages
                system=history_openai_format[0]["content"],
                model=state["model"],
                temperature=0,
            ) as stream:
                for text in stream.text_stream:
                    partial_message += text

                    partial_message = handle_latex_delimeter(partial_message)
                    yield partial_message
        except Exception as e:
            raise gr.Error(f"An error occurred: {e}")

    elif state["model"] in ["llama-3-1-70b", "llama-3-1-8b", "phi-3-small", "phi-3-medium"]:
        open_source_client = None
        if state["model"] == "llama-3-1-70b":
            open_source_client = llama_3_1_70b_client
        elif state["model"] == "llama-3-1-8b":
            open_source_client = llama_3_1_8b_client
        elif state["model"] == "phi-3-medium":
            open_source_client = phi_3_medium_client
        elif state["model"] == "phi-3-small":
            open_source_client = phi_3_small_client
        
        history_open_source_format = []
        
        for message in history_openai_format:
            if message["role"] == "system":
                history_open_source_format.append(SystemMessage(content=message["content"]))
            elif message["role"] == "assistant":
                history_open_source_format.append(AssistantMessage(content=message["content"]))
            elif message["role"] == "user":
                history_open_source_format.append(UserMessage(content=message["content"]))
                
        try:
            response = open_source_client.complete(
                stream=True,
                temperature=0,
                max_tokens=2000,
                messages=history_open_source_format
            )
            partial_message = "" 
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    partial_message = partial_message + chunk.choices[0].delta.content

                    partial_message = handle_latex_delimeter(partial_message)
                    yield partial_message
                
        except Exception as e:
            raise gr.Error(f"An error occurred: {e}")
    else:
        raise ValueError("Invalid model")

def problem_bank_click(prompt_bank_clicked_problem, solved_problems):
    if int(prompt_bank_clicked_problem[1]) in solved_problems:
        raise gr.Error("You have already tutored this problem. Please select another problem from the Math Problem Bank.", duration=5)
    return prompt_bank_clicked_problem[2]

def problem_bank_update(problem_search, difficulty_level, problem_type):
    # Initialize the filter to select all rows
    filter_condition = pd.Series(True, index=math_data_concise.index)
    
    # Add conditions based on the difficulty level
    if difficulty_level != "All":
        filter_condition &= (math_data_concise['level'] == difficulty_level)
    
    # Add conditions based on the problem type
    if problem_type != "All":
        filter_condition &= (math_data_concise['type'] == problem_type)
    
    # Add condition based on the problem search term
    if problem_search.strip():
        filter_condition &= (math_data_concise['problem'].str.contains(problem_search, case=False, na=False))
    
    # Apply the filter condition
    new_problem_list = math_data_concise[filter_condition].values.tolist()
    new_problem_list = [[str(item) for item in row] for row in new_problem_list]
    
    # Return the filtered dataset
    return gr.Dataset(samples=new_problem_list)


def finish_conversation():
    return gr.update(visible=True)

def textbox_submit(turn_count):
    return (turn_count[0] + 1, turn_count[0]), gr.update(visible=False)

def ask_tutor_click(problem_bank, sketch_pad, why_ask_tutor, why_ask_tutor_other_box, math_expertise,  state, solved_problems):
    if not problem_bank:
        raise gr.Error("Please select a problem from the Math Problem Bank before asking the tutor.", duration=5)
    state["why_ask_tutor"] = why_ask_tutor
    if why_ask_tutor == "Other reason":
        state["why_ask_tutor_other_reason"] = why_ask_tutor_other_box
    
    if state["why_ask_tutor"] in ["L3", "L4"]:
        raise gr.Error("Please select problems that are more challenging, ones that you don't know how to solve from scratch and feel the need to engage with the tutor in multiple interactions.")

    if not state["why_ask_tutor"]:
        raise gr.Error("Please select how do you feel about this problem?", duration=5)
    if state["why_ask_tutor"] == "Other reason" and not state["why_ask_tutor_other_reason"]:
        raise gr.Error("Please write down the reason why you want to ask the tutor.", duration=5)
    
    if not math_expertise:
        raise gr.Error("Please select your math expertise level before asking the tutor.", duration=5)
    state["math_expertise"] = math_expertise
    state["turk_initial_solution"] = sketch_pad
    problem_id = int(problem_bank[1])
    state["problem_id"] = problem_id
    state["question"] = problem_bank[2]
    state["level"] = problem_bank[3]
    state["type"] = problem_bank[4]
    row = math_data.loc[math_data['problem_id'] == problem_id]
    state["solution"] = row.iloc[0]["solution"].strip()
    state["similar_question"] = row.iloc[0]["top_train_problem"].strip()

    if not state["model"]:
        model_list = ["phi-3-medium", "mistral-large-latest", "gpt-4-turbo", "phi-3-small", "claude-3-5-sonnet-20240620",
                      "gpt-4o", "gpt-4o-mini", "llama-3-1-70b", "llama-3-1-8b"]
        
        # Setting equal probabilities
        probabilities = [1/len(model_list)] * len(model_list)

        # randomly sample a model
        state["model"] = random.choices(model_list, probabilities)[0]

    solved_problems.append(problem_id)
    solved_problems = list(set(solved_problems))

    state["solved_problems"] = sorted(solved_problems)
    state["problem_1_start_time"] = time.time()
    state["select_problem_time"] = state["problem_1_start_time"] - state["start_time"]

    return gr.update(visible=True), gr.update(visible=True), state["question"], \
            gr.update(visible=False), gr.update(visible=False), sketch_pad, state, solved_problems

def check_answer_click(answer, state):
    solution = state["solution"]
    pattern = r"\\boxed{((?:[^{}]+|{[^{}]*})*)}"
    match = re.search(pattern, solution)
    correct_answer = match.group(1)
    
    prompt_template = """"You are a math expert. Your task is to evaluate whether the student's answer matches the correct answer. Note that in mathematics, answers can be expressed in various formats and may include LaTeX notation. Determine the correctness of the student's answer based on its equivalence to the correct answer. Output "Correct" if the answer is correct, otherwise output "Incorrect.".

## Note: it's okay that the student doesn't include the base, as long as the number is correct.

## Output format:
You should only output either Correct or Incorrect, nothing else.
    
# Question: {question}
# Correct Answer: {correct_answer}
# Student's Answer: {student_answer}
# Correct or Incorrect:"""

    prompt = prompt_template.format(question=state["question"], correct_answer=correct_answer, student_answer=answer)
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=100,
        n=1
    )

    if "incorrect" in response.choices[0].message.content.lower():
        return "Sorry, you answer is not correct!", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
    else:
        return "Great job, you answer is correct!", gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False)
        
def solve_or_not(solve_or_not):
    if solve_or_not == "yes":
        return gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False)
    else:
        return gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=True)

def similar_solve_or_not():
    return gr.update(interactive=True), gr.update(interactive=True)

def other_box_update(why_stop_radio):
    if why_stop_radio == "Other reason":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)
    
def why_tutor_other_box_update(why_ask_tutor):
    if why_ask_tutor == "Other reason":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)
    
def check_solution(question, solution):
    prompt_template = """"Evaluate whether the student's solution is spam. Do not check for correctness, just check whether they make a reasonable effort to solve the problem.

Output a brief reasoning process first, then at the end, output "Spam" or "Not Spam," where "Spam" indicates that the solution is spam and "Not Spam" indicates that the student made a good effort.

# Examples of Spam:
- Completely unrelated content
- Random characters or symbols
- Repetitive, nonsensical text

# Note: It is okay if the student couldn't solve the problem, as long as they made a reasonable effort. Be lenient in your evaluation.

# Question: {question}
# Student's solution: {solution}
# Brief reasoning on whether spam:"""

    prompt = prompt_template.format(question=question, solution=solution)

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=1000,
        n=1
    )

    return response.choices[0].message.content

def submit_click(turk_final_answer, turk_solution, final_solve_or_not, chatbot,
                        strength_weakness_box, why_stop_radio, 
                        why_stop_other_box, overall_rating, feedback_box, state):
    state["turk_final_answer"] = turk_final_answer
    state["turk_solution"] = turk_solution
    state["solve_or_not"] = final_solve_or_not
    user_queries = [turn[0] for turn in chatbot]
    ai_responses = [turn[1] for turn in chatbot]
    state["ai_responses"] = ai_responses
    state["user_queries"] = user_queries
    state["strength_weakness"] = strength_weakness_box
    state["why_stop"] = why_stop_radio
    if why_stop_radio == "Other reason":
        state["why_stop_other_reason"] = why_stop_other_box
    if overall_rating >= 1 and overall_rating <= 10:
        state["overall_rating"] = overall_rating
    else:
        state["overall_rating"] = 0
    state["optional_feedback"] = feedback_box
    state["problem_1_end_time"] = time.time()
    state["problem_1_time_spent"] = state["problem_1_end_time"] - state["problem_1_start_time"]
    
    # checking constraints
    if not state.get("cheat", ""):
        if not state["ai_responses"]:
            raise gr.Error("Please chat with the AI tutor with as you need to make reasonable efforts to solve the problem before submitting the HIT.", duration=5)
    
        if not state["strength_weakness"]:
            raise gr.Error("Please write down the strengths and weaknesses of the tutor before submitting the HIT.", duration=5)
        if not state["why_stop"]:
            raise gr.Error("Please select why you end the tutoring session before submitting the HIT.", duration=5)
        if state["why_stop"] == "Other reason" and not state["why_stop_other_reason"]:
            raise gr.Error("Please write down the reason why you end the tutoring session before submitting the HIT.", duration=5)
        if not state["overall_rating"]:
            raise gr.Error("Please rate the overall tutoring experience from 1 to 10 before submitting the HIT.", duration=5)
        
        if not state["turk_solution"] or len(state["turk_solution"].split()) < 10:
            raise gr.Error("Please make a reasonable effort to clearly articulate your solution in at least two sentences, ensuring others can understand your reasoning, to submit the HIT.", duration=5)
        else:
            spam_or_not = check_solution(state["question"], state["turk_solution"])

            if "not spam" not in spam_or_not.lower():
                logger.info(f"Spam solution: {state['turk_solution']}")
                raise gr.Error(f"Please make a reasonable effort to clearly articulate your solution in at least two sentences, ensuring others can understand your reasoning, to submit the HIT. More detail: {spam_or_not}", duration=15)
    
    if not state["assignmentId"]:
        gr.Info("Thank you for your participation! Your response has been submitted. Jumping to the next one", duration=2)
        file_path = os.path.join(FOLDER_PATH, state["username"], f"{state['user_id']}_{state['problem_id']}.json")
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=4)
        
        hf_api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"{state['username']}/{state['user_id']}_{state['problem_id']}.json",
            repo_id=DATASET_REPO_URL,
            repo_type="dataset",
        )
        time.sleep(2)
    else:
        gr.Info("Thank you for your participation! Your response is being submitted", duration=2)
        file_path = os.path.join(FOLDER_PATH, state["username"], f"{state['assignmentId']}_{state['problem_id']}.json")
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=4)

        hf_api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"{state['username']}/{state['assignmentId']}_{state['problem_id']}.json",
            repo_id=DATASET_REPO_URL,
            repo_type="dataset",
        )
        time.sleep(2)

    if state["username"] == "mturk":
        logger.info(f"<><>submit AssignmentId: {state['assignmentId']}, WorkerId: {state['workerId']}, ProblemId: {state['problem_id']}, has been submitted.")
    else:
        logger.info(f"<><>submit Username: {state['username']}, ProblemId: {state['problem_id']}, has been submitted.")

    return state

def go_to_next_problem(state_dict):
    state_dict["moved_to_problem_2"] = True
    state_dict["problem_1_turns"] = len(state_dict["ai_responses"])
    state_dict["problem_1_end_time"] = time.time()
    state_dict["problem_1_time_spent"] = state_dict["problem_1_end_time"] - state_dict["problem_1_start_time"]
    return state_dict, state_dict["similar_question"], gr.update(visible=True), gr.update(visible=False), \
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),\
                      gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)


def go_to_next_problem_click(turk_final_answer, turk_solution, final_solve_or_not, chatbot,
                                                       strength_weakness_box, why_stop_radio, why_stop_other_box, overall_rating, state):
    state["turk_final_answer"] = turk_final_answer
    state["turk_solution"] = turk_solution
    state["solve_or_not"] = final_solve_or_not
    user_queries = [turn[0] for turn in chatbot]
    ai_responses = [turn[1] for turn in chatbot]
    state["ai_responses"] = ai_responses
    state["user_queries"] = user_queries
    state["strength_weakness"] = strength_weakness_box
    state["why_stop"] = why_stop_radio
    if why_stop_radio == "Other reason":
        state["why_stop_other_reason"] = why_stop_other_box
    if overall_rating >= 1 and overall_rating <= 10:
        state["overall_rating"] = overall_rating
    else:
        state["overall_rating"] = 0
    
    # checking constraints
    if not state.get("cheat", ""):
        if not state["ai_responses"]:
            raise gr.Error("Please chat with the AI tutor as you need to make reasonable efforts before going to the Problem 2.", duration=5)

        if not state["strength_weakness"]:
            raise gr.Error("Please write down the strengths and weaknesses of the tutor before going to the Problem 2.", duration=5)
        if not state["why_stop"]:
            raise gr.Error("Please select why you end the tutoring session before going to the Problem 2.", duration=5)
        if state["why_stop"] == "Other reason" and not state["why_stop_other_reason"]:
            raise gr.Error("Please write down the reason why you end the tutoring session before going to the Problem 2.", duration=5)
        if not state["overall_rating"]:
            raise gr.Error("Please rate the overall tutoring experience from 1 to 10 before going to the Problem 2.")
        
        if not state["turk_solution"] or len(state["turk_solution"].split()) < 10:
            raise gr.Error("Please make a reasonable effort to clearly articulate your solution in at least two sentences, ensuring others can understand your reasoning, before going to the Problem 2.", duration=5)
        else:
            spam_or_not = check_solution(state["question"], state["turk_solution"])

            if "not spam" not in spam_or_not.lower():
                logger.info(f"Spam solution: {state['turk_solution']}")
                raise gr.Error(f"Please make a reasonable effort to clearly articulate your solution in at least two sentences, ensuring others can understand your reasoning, before going to the Problem 2. More detail: {spam_or_not}", duration=15)
    return state
    

def submit_similar_click(similar_turk_final_answer, similar_turk_solution, similar_solve_or_not, chatbot, feedback_box, 
                         strength_weakness_box_for_similar, why_stop_radio_for_similar, why_stop_other_box_for_similar,
                                    overall_rating_for_similar, state):
    state["turk_problem_2_final_answer"] = similar_turk_final_answer
    state["turk_problem_2_solution"] = similar_turk_solution
    state["problem_2_solve_or_not"] = similar_solve_or_not
    user_queries = [turn[0] for turn in chatbot]
    ai_responses = [turn[1] for turn in chatbot]
    state["ai_responses"] = ai_responses
    state["user_queries"] = user_queries
    state["optional_feedback"] = feedback_box
    state["strength_weakness_for_problem_2"] = strength_weakness_box_for_similar
    state["why_stop_for_problem_2"] = why_stop_radio_for_similar
    if why_stop_radio_for_similar == "Other reason":
        state["why_stop_other_reason_for_problem_2"] = why_stop_other_box_for_similar
    if overall_rating_for_similar >= 1 and overall_rating_for_similar <= 10:
        state["overall_rating_for_problem_2"] = overall_rating_for_similar        

    state["problem_2_end_time"] = time.time()
    state["problem_2_time_spent"] = state["problem_2_end_time"] - state["problem_1_end_time"]
    
    # check constraints
    if not state.get("cheat", ""):
        if not state["turk_problem_2_solution"] or len(state["turk_problem_2_solution"].split()) < 10:
            raise gr.Error("Please make a reasonable effort to clearly articulate your solution in at least two sentences, ensuring others can understand your reasoning, to submit the HIT.", duration=5)
        else:
            spam_or_not = check_solution(state["similar_question"], state["turk_problem_2_solution"])

            if "not spam" not in spam_or_not.lower():
                logger.info(f"Spam solution: {state['turk_solution']}")
                raise gr.Error(f"Please make a reasonable effort to clearly articulate your solution in at least two sentences, ensuring others can understand your reasoning, to submit the HIT. More detail: {spam_or_not}", duration=15)

    if not state["assignmentId"]:
        gr.Info("Thank you for your participation! Your response has been submitted. Jumping to the next one", duration=2)
        file_path = os.path.join(FOLDER_PATH, state["username"], f"{state['user_id']}_{state['problem_id']}.json")
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=4)

        hf_api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"{state['username']}/{state['user_id']}_{state['problem_id']}.json",
            repo_id=DATASET_REPO_URL,
            repo_type="dataset",
        )
        time.sleep(2)
    else:
        gr.Info("Thank you for your participation! Your response is being submitted", duration=2)
        file_path = os.path.join(FOLDER_PATH, state["username"], f"{state['assignmentId']}_{state['problem_id']}.json")
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=4)

        hf_api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"{state['username']}/{state['assignmentId']}_{state['problem_id']}.json",
            repo_id=DATASET_REPO_URL,
            repo_type="dataset",
        )
        time.sleep(2)

    if state["username"] == "mturk":
        logger.info(f"<><>submit_similar AssignmentId: {state['assignmentId']}, WorkerId: {state['workerId']}, ProblemId: {state['problem_id']}, has been submitted.")
    else:
        logger.info(f"<><>submit_similar Username: {state['username']}, ProblemId: {state['problem_id']}, has been submitted.")
    return state

tachyon_head = '<link rel="stylesheet" href="https://unpkg.com/tachyons@4.12.0/css/tachyons.min.css"/>'

with gr.Blocks(delete_cache=(60, 3600),
                fill_height=True,
               title="Math Tutoring",
               head=tachyon_head,
               css="#chatbot {flex-grow: 1 !important; overflow: auto !important;}"
               "#turn-level-rating { height: calc(100vh - 800px) !important; overflow-y: auto !important; overflow-x: hidden !important;}"
               "footer {visibility: hidden;}"
               ".f4-5 {font-size: 1.05rem;}"
               "hr {margin-top: 0.5em; border: none; height: 1.2px; color: #333;  /* old IE */ background-color: #333;  /* Modern Browsers */}"
               ".gap {gap: 6px}"
               "#ask-tutor-button {color: #357EDD}"
               "#check-answer-button {color: #FF4136}" # red
               "#finish-tutoring-button {color: #357EDD}" # blue 
                "#change-problem-button {color: #19A974}" # green
                "#optional-feedback .svelte-1w6vloh .svelte-1w6vloh {font-size: 1.05rem; font-weight: 550; color: #e5a400}" # gold
                "#problem-bank-solved-accordion {margin-top: 10px}"
                "#problem-bank-solved-accordion .svelte-1w6vloh .svelte-1w6vloh {font-size: 1.05rem; font-weight: 550;}"
                "#problem-bank .paginate,  #problem-bank .label {font-size: 1.05rem; color: #000000}"
                "#problem-bank .paginate button.svelte-p5q82i {margin-right: 0.5em; margin-left: 0.5em;}"
                "#problem-bank-solved .paginate,  #problem-bank-solved .label {font-size: 1.05rem; color: #000000}"
                "#problem-bank-solved .paginate button.svelte-p5q82i {margin-right: 0.5em; margin-left: 0.5em;}"
                "#control-bar {width: 60%; margin: 0 auto;}"
                "#selected-problem {border: 1.5px solid black !important; border-radius: .5rem; padding-top: 0.5rem; padding-bottom: 0.5rem; padding-left: 1rem; padding-right: 1rem;}"
                "#selected-problem span {font-size: 1rem !important;}"
                "#instance-description {border: 1.5px solid black !important; border-radius: .5rem; padding-top: 0.5rem; padding-bottom: 0.5rem; padding-left: 1rem; padding-right: 1rem;}"
                "#instance-description span {font-size: 1rem !important;}"
                "#similar-instance-description {border: 1.5px solid #19A974 !important; border-radius: .5rem; padding-top: 0.5rem; padding-bottom: 0.5rem; padding-left: 1rem; padding-right: 1rem;}"
                "#similar-instance-description span {font-size: 1rem !important;}"
            ) as demo:
    
    user_id = gr.JSON(visible=False)
    solved_problems = persist(gr.State(value=[]), cookies)
    state = gr.JSON(visible=False)
    turn_count = gr.State([0, 0]) # new value, old value
    turker_instruction = gr.HTML('''<p class="f4-5"><b>General Instructions:</b> In this task, you will choose some math problems that you are unsure how to solve but want to learn. Start by selecting a problem from the <b>Math Problem Bank</b>, which contains around 1,000 problems.

    Next, you can ask an AI tutor for help which will open the <span class="b" style="color: #357EDD">Tutoring Window</span>, where you can chat with the AI tutor for guidance. Btw, please <span style="color: red">don't use dark mode</span> as it make the webpage unreadable.
    <span class="i" style="display: block;">Note: **The annotations collected in this HIT will be publicly released for research purposes. No personally identifiable information will be included.**</span>
    </p>
    ''')
    turker_bonus_instruction = gr.HTML("""
        <p class="f4-5 i">
            Note: **we value your dedication! Bonuses will be awarded based on the effort demonstrated in your submission. We will review your work to assess the effort involved.**
        </p>
    """)


    with gr.Column(elem_classes=["mt2"]) as problem_bank_col:
        math_data_list = math_data_concise.values.tolist()
        
        unique_levels_list = math_data_concise["level"].unique().tolist()
        unique_levels_list = ["All"] + unique_levels_list

        unique_types_list = math_data_concise["type"].unique().tolist()
        unique_types_list = ["All"] + unique_types_list
        math_data_list = [[str(item) for item in row] for row in math_data_list]
        math_data_problem_markdown = gr.Markdown(latex_delimiters=latex_delimeter_set, visible=False)
        with gr.Column(elem_id = "control-bar") as control_bar:
            with gr.Row() as search_bar :
                problem_search = gr.Textbox(placeholder="Keyword Search", scale=4, show_label=False)
                search_button = gr.Button(value="Search", elem_id="search-button", scale=1)
            with gr.Row() as filter_bar:
                difficulty_level_dropdown = gr.Dropdown(label="Difficulty Level", choices=unique_levels_list, value="All")
                problem_type_dropdown = gr.Dropdown(label="Problem Type", choices=unique_types_list, value="All")
        
        problem_bank = gr.Dataset(components=["markdown", "markdown", math_data_problem_markdown, "markdown", "markdown"], 
                   samples=math_data_list, headers=["Index", "Problem ID", "Problem", "Difficulty Level", "Type"],
                   label="Math Problem Bank", elem_id="problem-bank", samples_per_page=10)

        gr.HTML(f"""<div class="br3 ph3 pv2 f4-5" style="background-color: #fbe0e0; margin-top: 7pt">
                    <b>Instruction:</b> Select a problem that you are unsure how to solve but interested in learning it.
                    </div>""")
        
        with gr.Accordion("Problems that you have asked AI tutor:", 
                          elem_id="problem-bank-solved-accordion") as problem_bank_solved_accordion:
            solved_problem_markdown = gr.Markdown(latex_delimiters=latex_delimeter_set, visible=False)
            solved_problems_data =  gr.Dataset(components=["markdown", "markdown", solved_problem_markdown, "markdown", "markdown"],
                                                headers=["Index", "Problem ID", "Problem", "Difficulty Level", "Type"],
                                                label="Tutored Problems" ,samples_per_page=5, elem_id="problem-bank-solved")
            
        gr.HTML("""<p class="f4-5 pv0" style="margin-top: 7pt !important; margin-bottom: 0 !important"><b>Selected problem</b>
                    (<span style="color: red">New:</span> please select problems that are <b>challenging</b> to you, ones that you don't know how to solve and feel the need to engage with the tutor in <b>multiple interactions.</b>)
                </p>""")
        selected_problem = gr.Markdown(visible=True, latex_delimiters=latex_delimeter_set, value=
                                        """... Click problem to select ...""",
                                        elem_id="selected-problem")
        gr.HTML(f"""
                <p class="f4-5 pv0 b" style="margin-top: 7pt !important; margin-bottom: 0 !important">Sketch Pad</p>
                <p class="f5 i">
                            If you want to write some steps first before asking the AI tutor, you can type below.
                    </p>""")
        sketch_pad= gr.Textbox(visible=True, lines=5, show_label=False)

        with gr.Row() as question_row:
            with gr.Column() as why_tutor_col:
                    gr.HTML("""
                            <p class="f4-5 pv0 b" style="margin-top: 7pt !important; margin-bottom: 0 !important">
                            Understanding your challenge</p>
                            <p class="f5 i" style="margin-bottom: 0 !important">How do you feel about this problem?</p>
                            <p class="f5 mt0"> <span style="color: red">New:</span> please select problems that are challenging (not the 3rd and 4th options).</p>""")
                    why_ask_tutor = gr.Radio(choices=[("1. I'm completely lost and don't understand the problem", "L1"), 
                                                       ("2. I understand the problem, but I don't know where to start", "L2"), 
                                                       ("3. I've written some steps, but now I'm stuck", "L3"), 
                                                       ("4. I can solve the problem, but I'm not confident in my solution", "L4"), 
                                                       ("Other reason")], 
                                                       show_label=False, interactive=True)
                    why_ask_tutor_other_box = gr.Textbox(lines=1, label="Wrie down your reason", visible=False)
                

            with gr.Column() as math_expertise_col:
                    gr.HTML("""
                            <p class="f4-5 pv0 b" style="margin-top: 7pt !important; margin-bottom: 0 !important">Math expertise</p>
                            <p class="f5 i">What's the highest-level of math coursework that you have attended?</p>""")
                    math_expertise = gr.Radio(choices=[("Elementary School (Grades K-5)", "k-5"), ("Middle School (Grades 6-8)", "6-8"), ("High School (Grades 9-12)","9-12"), ("Undergraduate-level", "undergrad"), ("Graduate-level", "grad"), ("Conduct studies or research", "research")], show_label=False, interactive=True)
            
        
        start_tutoring_button = gr.Button(value="Ask Tutor", elem_id="ask-tutor-button", scale=1)

    with gr.Row(elem_classes=["mt2"], visible=False) as main_row:
        with gr.Column(scale=2, elem_classes=["ba pa2 bw1 b--black-60 br3"]) as first_part:
            gr.HTML("""<p class='f4-5 mb0'><span class="b pa1" style="border-left: 2px solid; border-bottom: 2px solid">Problem Window</span></p>""")
            problem_window_instruction = gr.HTML(f"""<div class="br3 ph3 pv2 f4-5" style="background-color: #e0e0eb">
                    <b>Instructions:</b> You will continue working on your solution of the problem in this window while you are interacting with the AI tutor.                                      
                    <ul class="f4-5 list">
                    <li style="margin-bottom:2px; margin-top:2px">If you believe you have successfully solved the problem, click 
                                                 <i style="color: #19A974">Go to Problem 2</i> to continue with a second problem that involved similar concepts to test your understanding.</li>
                    <li>If you are unable to solve the problem, click <i>Submit the HIT</i> to complete this task.</li>
                    </ul>
                    <p><b>Note:</b> Please make a reasonable effort to clearly articulate your solution in at least two sentences, ensuring others can understand your reasoning. Even if you cannot fully solve the problem, write down the steps you managed to complete.</p>
                    </div>""")
            
            instance_description = gr.Markdown(visible=True, latex_delimiters=latex_delimeter_set,
                                        elem_id="instance-description")
            
            with gr.Row():
                with gr.Column(scale=5):
                    turk_solution = gr.Textbox(lines=5, label="Your Step-by-step Solution", placeholder="Write down your step-by-step solution here.")
                with gr.Column(scale=2):
                    turk_final_answer = gr.Textbox(lines=1, label="Your Final Answer", placeholder="Write down your final answer here.")

            with gr.Column() as final_button_col:
                final_solve_or_not = gr.Radio([("I think I solved the problem", "yes"), ("I couldn't solve the problem", "no")], label="Select whether you solved the problem or not")
                with gr.Row():
                    change_problem_button = gr.Button(value="Go to Problem 2", interactive=False, elem_id="change-problem-button")
                    submit_hit_button = gr.Button(value="Submit the Hit", visible=False, interactive=False)
                    submit_hf_button = gr.Button(value="Submit the Task", visible=False, interactive=False)

        with gr.Column(visible=False, scale=2, elem_classes=["ba pa2 bw1 b--black-60 br3"]) as similar_part:
            gr.HTML("""<p class='f4-5 mb0'><span class="b pa1" style="border-left: 2px solid; border-bottom: 2px solid">Problem Window</span></p>""")
            gr.HTML(f"""<div class="br3 ph3 pv2 f4-5" style="background-color: #b7f4de">
                    <b>Instructions:</b>
                    You are now moving onto Problem 2, which is based on the same concepts as the first problem, to further test your understanding.
                    You can also ask the AI tutor for help if needed, but <b> please try to solve the problem first before asking tutor.</b>
                    <p><b>Note:</b> Like the first problem, please make a reasonable effort to clearly articulate your solution in at least two sentences, ensuring others can understand your reasoning. Even if you cannot fully solve the problem, write down the steps you managed to complete.</p>
                    </div>""")
            similar_instance_description = gr.Markdown(latex_delimiters=latex_delimeter_set, 
                                                       elem_id="similar-instance-description") 
            
            with gr.Row():
                with gr.Column(scale=5):
                    similar_turk_solution = gr.Textbox(lines=5, label="Your Step-by-step Solution", placeholder="Write down your step-by-step solution here.")
                with gr.Column(scale=2):
                    similar_turk_final_answer = gr.Textbox(lines=1, label="Your Final Answer", placeholder="Write down your final answer here.")
            
            similar_solve_or_not_radio = gr.Radio([("I think I solved the problem", "yes"), ("I couldn't solve the problem", "no")], label="Select whether you solved the problem or not")

            submit_hf_button_similar = gr.Button(value="Submit the Task", visible=False, interactive=False)
            submit_hit_button_similar = gr.Button(value="Submit the Hit", visible=False, interactive=False)

        with gr.Column(visible=False, scale=3, elem_classes=["ba pa2 bw1 b--black-60 br3 b--blue"]) as tutoring_interface:
            gr.HTML("""<p class='f4-5 mb0'><span class="b pa1" style="border-left: 2px solid; border-bottom: 2px solid; color:#357EDD">Tutoring Window</span></p>""")
            with gr.Row():
                with gr.Column(scale=3, elem_id="col"):
                    problem_1_tutoring_instruction = gr.HTML(f"""<div class="br3 ph3 pv2" style="background-color: #cceeff"><p class='f4-5'>
                            <b>Instructions:</b> In this window, you can interact with the AI tutor for assistance.
                            You can ask any questions that might help you solve the problem such as requesting explanations of unfamiliar concepts,
                            seeking guidance on how to approach the problem, or getting clarification on any points of confusion. However, you <b>should not</b> ask the tutor to simply provide a solution, as the goal is
                             for you to learn the knowledge needed to solve the problem independently. </p>
                            <p class="f4-5 mt0 mb0"><b>Notes:</b> 1. The AI tutor already knows the problem statement, so you do not need to repeat it. 
                                                             However, the tutor doesn't know your solution, so you need to provide it if you want to show it to the tutor.</p>
                            <p class="f4-5 mt0 mb0">2. Just like human, the AI tutor might sometimes make mistakes, so double-check its response as needed.</p>
                            <p class="f4-5 mt0 mb0">3. Don't forget to answer the three survey questions after you finish the tutoring session.</p>
                                <div class="br3 ph3 pv2" style="background-color: white"><p class="f4-5 mt0 mb0">
                                <span style="color: red">New:</span> For AI's response in each turn,
                                please give a 👍 if it's good, a 👎 if it's bad, 
                                leave as it is if it's just okay. Ignore the thumbs for your own queries.</p></div>
                            <div>""")
                    problem_2_tutoring_instruction = gr.HTML(f"""<div class="br3 ph3 pv2" style="background-color: #cceeff"><p class='f4-5'>
                            <b>Instruction:</b> You can interact with the AI tutor for assistance on problem 2.
                            Like the first problem, you can ask any questions that might help you solve the problem such as requesting explanations of unfamiliar concepts,
                            seeking guidance on how to approach the problem, or getting clarification on any points of confusion. However, you <b>should not</b> ask the tutor to simply provide a solution, as the goal is
                             for you to learn the knowledge needed to solve the problem independently. </p>
                            <p class="f4-5 mt0 mb0"><b>Notes:</b> 1. The AI tutor already knows the problem statement, so you do not need to repeat it.
                                                             However, the tutor doesn't know your solution, so you need to provide it if you want to show it to the tutor.</p>
                            <p class="f4-5 mt0 mb0">2. Just like human, the AI tutor might sometimes make mistakes, so double-check its response as needed.</p>
                            <p class="f4-5 mt0 mb0">3. There are three optional survey questions about the tutoring experience for problem 2.</p>
                            <div class="br3 ph3 pv2" style="background-color: white"><p class="f4-5 mt0 mb0">
                                <span style="color: red">New:</span> For AI's response in each turn,
                                please give a 👍 if it's good, a 👎 if it's bad, 
                                leave as it is if it's just okay. Ignore the thumbs for your own queries.</p></div>
                                <div>
                            <div class="br3 ph3 pv2" style="background-color: white"><p class="f4-5 mt0 mb0">
                                <span style="color: red">New Note:</span> Since you've already solved problem 1 and gained some understanding, 
                                                             please try to solve problem 2 on your own first. Only ask the tutor for help if you truly need it.</p></div>
                                <div>""",
                            visible=False)
                    chat_interface=gr.ChatInterface(fn=echo, additional_inputs=[state], chatbot=gr.Chatbot(
                            label="AI Tutor",
                            render=False,
                            elem_id="chatbot",
                            avatar_images=(None, "img/ai.png"),
                            latex_delimiters=latex_delimeter_set,
                            likeable=True,
                        ), multimodal=True, retry_btn=None, clear_btn=None, undo_btn=None, autofocus=False, concurrency_limit=4)
                    chat_interface.chatbot.like(vote, inputs=state, outputs=state) 
                    
            finish_conversation_instruction = gr.HTML(f"""<div class="br3 ph3 pv2 f4-5" style="background-color: #cceeff">
                        Click <span class="i" style="color: #357EDD">Finish Tutoring</span> once you are able to solve the problem or if you find that the AI tutor is no longer helpful. 
                                        Then you will be asked to answer <b>three survey questions</b> regarding your overall tutoring experience.
                    </div>""")
            finish_conversation_button = gr.Button(value="Finish Tutoring", elem_id="finish-tutoring-button")

            with gr.Column(visible=False) as tutoring_annotation_column:
                gr.HTML("""
                        <p class="f4-5 b">Three survey questions about your tutoring experience:</p>
                        """)
                gr.HTML("""
                     <p class="f4-5">1. What are the strengths and weaknesses of the tutor?</p>
                    """)
                strength_weakness_box = gr.Textbox(lines=2, label="Write down few sentences on the strengths and weaknesses of the tutor", visible=True)
                
                gr.HTML("""
                        <p class="f4-5">2. Why did you end the tutoring session?</p>
                        """)

                why_stop_radio = gr.Radio(["I felt confident enough to solve the problem.", "I found the AI tutor not helpful.", "I encountered technical issues with the interface.", "Other reason"], show_label=False, visible=True)
                why_stop_other_box = gr.Textbox(lines=1, label="Wrie down your reason", visible=False)

                with gr.Row(elem_classes=["pa1"]) as overall_rating_row:
                    with gr.Column(scale=3):
                        gr.HTML(
                            """ 
                            <p class="f4-5">3. Rate your tutoring experience from 1 to 10 based on the following criteria:</p>
                            <ul class="f4-5 list">
                                <li>Score 1 ~ 2 <span class="b"> (very poor)</span>: The tutoring quality is very poor, lacks coherence, and fails to contribute to the my understanding of the problem.</li>
                                <li>Score 3 ~ 4 <span class="b"> (poor)</span>: The tutoring quality is poor, offering minimal help and not aiding my solving the problem much.</li>
                                <li>Score 5 ~ 6 <span class="b"> (average)</span>: The tutoring quality is average, but contains errors or omits important information, leading to partial understanding or confusion.</li>
                                <li>Score 7 ~ 8 <span class="b"> (good)</span>:  The tutoring quality is good, providing useful information that aids the my understanding and problem-solving effectively.</li>
                                <li>Score 9 ~ 10 <span class="b"> (very good)</span>: The tutoring quality is very good, offering clear, comprehensive, and insightful guidance that significantly enhances the my understanding and problem-solving abilities.</li>
                            </ul>
                            """
                        )
                    with gr.Column(scale=1):
                        overall_rating = gr.Slider(value=0, label="Overall Rating", minimum=1, maximum=10, step=1, 
                                                info="rate from 1 (worst) to 10 (best)",interactive=True)
                        
            with gr.Column(visible=False) as survey_questions_for_similar:
                survey_instruction_for_similar = gr.HTML(f"""<div class="br3 ph3 pv2 f4-5" style="background-color: #dcffcc">
                        Like the first problem, there are three survey questions (but optional) about your tutoring experience on <span style="color: #19A974">problem 2</span> if you chatted with the AI tutor.
                        Please share any new findings you have while chatting with the AI tutor on <span style="color: #19A974">problem 2</span>.
                    </div>""")

                with gr.Column(visible=True) as tutoring_annotation_column_for_similar:
                    gr.HTML("""
                            <p class="f4-5 b">Three optional survey questions about your tutoring experience on <span style="color: #19A974">problem 2</span>:</p>
                            """)
                    gr.HTML("""
                        <p class="f4-5">[Optional] 1. What are the <b>new</b> strengths and weaknesses of the tutor that you found?</p>
                        """)
                    strength_weakness_box_for_similar = gr.Textbox(lines=2, label="Write down few sentences on the strengths and weaknesses of the tutor", visible=True)
                    
                    gr.HTML("""
                            <p class="f4-5">[Optional] 2. Why did you end the tutoring session?</p>
                            """)

                    why_stop_radio_for_similar = gr.Radio(["I felt confident enough to solve the problem.", "I found the AI tutor not helpful.", "I encountered technical issues with the interface.", "Other reason"], show_label=False, visible=True)
                    why_stop_other_box_for_similar = gr.Textbox(lines=1, label="Wrie down your reason", visible=False)

                    with gr.Row(elem_classes=["pa1"]) as overall_rating_row_for_similar:
                        with gr.Column(scale=3):
                            gr.HTML(
                                """ 
                                <p class="f4-5">[Optional] 3. Rate your tutoring experience from 1 to 10 based on the following criteria:</p>
                                <ul class="f4-5 list">
                                    <li>Score 1 ~ 2 <span class="b"> (very poor)</span>: The tutoring quality is very poor, lacks coherence, and fails to contribute to the my understanding of the problem.</li>
                                    <li>Score 3 ~ 4 <span class="b"> (poor)</span>: The tutoring quality is poor, offering minimal help and not aiding my solving the problem much.</li>
                                    <li>Score 5 ~ 6 <span class="b"> (average)</span>: The tutoring quality is average, but contains errors or omits important information, leading to partial understanding or confusion.</li>
                                    <li>Score 7 ~ 8 <span class="b"> (good)</span>:  The tutoring quality is good, providing useful information that aids the my understanding and problem-solving effectively.</li>
                                    <li>Score 9 ~ 10 <span class="b"> (very good)</span>: The tutoring quality is very good, offering clear, comprehensive, and insightful guidance that significantly enhances the my understanding and problem-solving abilities.</li>
                                </ul>
                                """
                            )
                        with gr.Column(scale=1):
                            overall_rating_for_similar = gr.Slider(value=0, label="Overall Rating", minimum=1, maximum=10, step=1, 
                                                    info="rate from 1 (worst) to 10 (best)",interactive=True)
    
    with gr.Accordion("Optional feedback on the interface or the task", open=False, elem_id="optional-feedback"):
        with gr.Column():
            feedback_textbox = gr.Textbox(lines=5, label="""Please leave any feedback or comments you have about the interface or the task here. Let us know how we can improve.""", visible=True)

    output = gr.Textbox(lines=2, label="AI Response", visible=False)

    # problem_bank_control_bar
    problem_bank.click(problem_bank_click, inputs=[problem_bank, solved_problems], outputs=[selected_problem])
    
    gr.on(
        triggers=[difficulty_level_dropdown.select, problem_type_dropdown.select, search_button.click],
        fn=problem_bank_update,
        inputs=[problem_search, difficulty_level_dropdown, problem_type_dropdown], 
        outputs=[problem_bank],
    )

    why_ask_tutor.select(why_tutor_other_box_update, inputs=[why_ask_tutor], outputs=[why_ask_tutor_other_box])

    why_stop_radio.select(other_box_update, inputs=[why_stop_radio], outputs=[why_stop_other_box])

    why_stop_radio_for_similar.select(other_box_update, inputs=[why_stop_radio_for_similar], 
                                      outputs=[why_stop_other_box_for_similar])

    finish_conversation_button.click(finish_conversation, outputs=[tutoring_annotation_column])

    final_solve_or_not.input(solve_or_not, final_solve_or_not, outputs=[change_problem_button, submit_hit_button, submit_hf_button])
    similar_solve_or_not_radio.input(similar_solve_or_not, outputs=[submit_hit_button_similar, submit_hf_button_similar])

    chat_interface.textbox.submit(textbox_submit, turn_count, outputs=[turn_count, tutoring_annotation_column])

    post_hit_js = """
        function(state) {
            // If there is an assignmentId, then the submitter is on mturk
            // and has accepted the HIT. So, we need to submit their HIT.
            
            const form = document.createElement('form');
            const turkSubmitTo = state.turkSubmitTo || "https://www.mturk.com";
            form.action = `${turkSubmitTo}/mturk/externalSubmit`;
            form.method = 'post';
            for (const key in state) {
                const hiddenField = document.createElement('input');
                hiddenField.type = 'hidden';
                hiddenField.name = key;
                hiddenField.value = state[key];
                form.appendChild(hiddenField);
            };
            document.body.appendChild(form);
            form.submit();
            return state;
        }
        """
    
    refresh_webpage_js = """
        function(state) {
            // Parse the URL parameters
            const urlParams = new URLSearchParams(window.location.search);
            
            // Construct the new URL
            const newUrl = window.location.origin + window.location.pathname + '?' + urlParams.toString();
            
            // Redirect to the new URL
            window.location.href = newUrl;
            
            return state;
        }
    """

    remove_problem_bank_js = """
        function(state) {
            var problemBank = document.getElementById("problem-bank");
            if (problemBank) {
                problemBank.remove();
            }
            return state;
        }
        """

    start_tutoring_button.click(ask_tutor_click, inputs = [problem_bank, sketch_pad, why_ask_tutor, why_ask_tutor_other_box, math_expertise, state, solved_problems], 
                                outputs=[tutoring_interface, main_row, instance_description, question_row, 
                                         problem_bank_col, turk_solution, state, solved_problems]).success(
                                             lambda state: state, inputs=[state], outputs=[state],
                                             js=remove_problem_bank_js
                                         )
    
    ##  submit to huggingface
    submit_hf_button.click(submit_click, inputs=[turk_final_answer, turk_solution, final_solve_or_not, chat_interface.chatbot,
                                strength_weakness_box, why_stop_radio, why_stop_other_box, overall_rating, feedback_textbox, state],
                                outputs=[state]).success(lambda state: state, inputs=[state], outputs=[state], js=refresh_webpage_js)
    
    submit_hf_button_similar.click(submit_similar_click, inputs=[similar_turk_final_answer, similar_turk_solution, 
                                    similar_solve_or_not_radio, chat_interface.chatbot, feedback_textbox, strength_weakness_box_for_similar, why_stop_radio_for_similar, why_stop_other_box_for_similar,
                                    overall_rating_for_similar, state], 
                                    outputs=[state]).success(lambda state: state, inputs=[state], outputs=[state], js=refresh_webpage_js)
    
    ## submit to mturk
    submit_hit_button.click(submit_click, inputs=[turk_final_answer, turk_solution, final_solve_or_not, chat_interface.chatbot,
                                strength_weakness_box, why_stop_radio, why_stop_other_box, overall_rating, feedback_textbox, state],
                                outputs=[state]).success(lambda state: state, inputs=[state], outputs=[state], js=post_hit_js)
    submit_hit_button_similar.click(submit_similar_click, inputs=[similar_turk_final_answer, similar_turk_solution,
                                    similar_solve_or_not_radio, chat_interface.chatbot, feedback_textbox, strength_weakness_box_for_similar, why_stop_radio_for_similar, why_stop_other_box_for_similar,
                                    overall_rating_for_similar, state],
                                    outputs=[state]).success(lambda state: state, inputs=[state], outputs=[state], js=post_hit_js)
    

    change_problem_button.click(go_to_next_problem_click, inputs=[turk_final_answer, turk_solution, final_solve_or_not, chat_interface.chatbot,
                                                       strength_weakness_box, why_stop_radio, why_stop_other_box, overall_rating, state], outputs=[state]).success(go_to_next_problem, inputs=[state], outputs=[state, similar_instance_description, 
                                    similar_part, first_part,
                                    finish_conversation_instruction,
                                    finish_conversation_button, tutoring_annotation_column,
                                    problem_1_tutoring_instruction, problem_2_tutoring_instruction, survey_questions_for_similar])
    
    cookie_js = '''
        function(value){
            let user_id = value['user_id']; // Access the user_id from the value dictionary
            document.cookie = 'user_id=' + user_id + '; Path=/;  SameSite=None; Secure'; // this allows iframe like in amt
            return value;
        }
    '''

    def update_and_print_solved_problems(solved_problems):        
        # Return the filtered dataset
        if solved_problems:
            # Filter DataFrame
            solved_problems_df = math_data_concise[math_data_concise['problem_id'].isin(solved_problems)]

            # Sort the filtered DataFrame by problem_id
            solved_problems_df = solved_problems_df.sort_values(by='index')

            solved_problems_list = solved_problems_df.values.tolist()
            solved_problems_list = [[str(item) for item in row] for row in solved_problems_list]
            return gr.Dataset(samples=solved_problems_list), gr.update(visible=True)
        else:
            return gr.Dataset(samples=None), gr.update(visible=False)
            

    demo.load(load_instance, None, outputs=[state, user_id,
                                            submit_hf_button, submit_hf_button_similar, 
                                            submit_hit_button, submit_hit_button_similar, turker_bonus_instruction]).then(
                            lambda user_id: user_id, inputs=[user_id], js=cookie_js).then(
                                update_and_print_solved_problems, inputs=[solved_problems], 
                                outputs=[solved_problems_data, problem_bank_solved_accordion]
                            )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=5, max_size=5)
    demo.launch(max_threads=10)
