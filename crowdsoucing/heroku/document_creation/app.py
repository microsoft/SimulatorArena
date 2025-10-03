import gradio as gr
import os

import time

import re

import uuid

import random

import pandas as pd
from openai import OpenAI
import anthropic

from mistralai import Mistral

from huggingface_hub import HfApi

from logger_config import logger

import json

from utils import handle_latex_delimeter, persist, sanitize_filename, extract_json

from azure.ai.inference import ChatCompletionsClient
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

DATA_PATH = "data/intent_bank.csv"
intent_bank_data = pd.read_csv(DATA_PATH)


# read cookies.json
with open("cookies.json", "r") as f:
    cookies = json.load(f)
    
with open("worker_user_id_dict.json", "r") as f:
    worker_user_id_dict = json.load(f)


with open("document_tracking.txt", "r") as f:
    document_tracking_prompt_template = f.read()

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

    model = query_params.get("model", "")

    document_type = query_params.get("document_type", "")

    # for mturk user set user_id to their previous user_id in case they remove the cookies
    if query_params.get("workerId", "") in worker_user_id_dict:
        user_id = worker_user_id_dict[query_params.get("workerId", "")]
    else:
        if "user_id" in request.cookies and request.cookies["user_id"] != "null":
            user_id = request.cookies["user_id"]
        else:
            user_id = str(uuid.uuid4())
        
    start_time = time.time()

    if not model:
        model_list = ["mistral-large-2407", "phi-3-small", "phi-3-medium", "claude-3-5-sonnet-20240620", 
                    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "llama-3-1-70b", "llama-3-1-8b"]

        probabilities = [1/len(model_list)] * len(model_list)

        model = random.choices(model_list, probabilities)[0]

    state_dict = {
            "username": username,
            "user_id": user_id,
            "datapath": DATA_PATH,
            "assignmentId": query_params.get("assignmentId", ""),
            "hitId": query_params.get("hitId", ""),
            "workerId": query_params.get("workerId", ""),
            "turkSubmitTo": query_params.get("turkSubmitTo", ""),
            "cheat": query_params.get("cheat", ""),
            "user_queries": [],
            "ai_responses": [],
            "background": {},
            "start_time": start_time,
            "model": model,
        }
    
    starting_html = """
            <div style="text-align: center; padding: 50px;">
                <h1 style="font-size: 48px; margin-bottom: 20px;">Welcome to Our AI-Assisted Document Creation Task</h1>
                <p style="font-size: 22px; margin-bottom: 40px;">
                    In this task, you will interact with an AI writing assistant to create a document. 
                    This task is broken into three simple steps designed to guide you through the experience.
                </p>
                <div style="max-width: 800px; margin: 0 auto; text-align: left;">
                    <ol style="font-size: 20px; line-height: 1.6;">
                        <li>
                            <strong>Step 1: Select Document Type and Intent that you are interested in</strong><br>
                            Choose a document type from options of <em>Creative Writing</em>, <em>Blog Post</em>,
                            or <em>Email/Letter</em>, along with the intent.
                        </li>
                        <li>
                            <strong>Step 2: Pre-Writing Preparation</strong><br>
                            Jot down some pre-writing details for your selected intent, this is for you to organize your thoughts like some rough plan on what to write.
                        </li>
                        <li>
                            <strong>Step 3: Interact with the AI Writing Assistant</strong><br>
                            Talk to the AI to bring your document to life!
                        </li>
                    </ol>
                </div>
            </div>
        """
    select_intent_window_html = """
        <div style="text-align: center; padding: 0;">
            <h2 style="font-size: 32px; margin-bottom: 10px; margin-top: 0">Step 1</h2>
            <p style="font-size: 20px; margin-bottom: 0;">
                Choose an intent that you are interested in writing about, or add your own intent if you have a specific idea in mind.
            </p>
            <p style="font-size: 20px; color: #D44C4C; margin-top: 0; margin-bottom: 30px;" class="i">
                Note: The final document that you'll create with AI will be between 100 and 500 words, so your intent should not be too simple.
            </p>
        </div>
        """
    
    if document_type == "creative_writing":
        cw_btn_visible = gr.update(visible=True)
        bp_btn_visible = gr.update(visible=False)
        em_btn_visible = gr.update(visible=False)
        add_own_type_dd = gr.update(choices=[
                        ("Creative Writing", "creative writing"),
                    ])
        starting_html = """
            <div style="text-align: center; padding: 50px;">
                <h1 style="font-size: 48px; margin-bottom: 20px;">Welcome to Our AI-Assisted Document Creation Task</h1>
                <p style="font-size: 22px; margin-bottom: 40px;">
                    In this task, you will interact with an AI writing assistant to create a <b>creative writing</b> piece. 
                    This task is broken into three simple steps designed to guide you through the experience.
                </p>
                <div style="max-width: 800px; margin: 0 auto; text-align: left;">
                    <ol style="font-size: 20px; line-height: 1.6;">
                        <li>
                            <strong>Step 1: Select Intent that you are interested in</strong><br>
                            Choose an intent that you want to write a creative writing piece about.
                        </li>
                        <li>
                            <strong>Step 2: Pre-Writing Preparation</strong><br>
                            Jot down some pre-writing details for your selected intent, this is for you to organize your thoughts like some rough plan on what to write.
                        </li>
                        <li>
                            <strong>Step 3: Interact with the AI Writing Assistant</strong><br>
                            Talk to the AI to bring your document to life!
                        </li>
                    </ol>
                </div>
            </div>
        """
    elif document_type == "blog_post":
        cw_btn_visible = gr.update(visible=False)
        bp_btn_visible = gr.update(visible=True)
        em_btn_visible = gr.update(visible=False)
        add_own_type_dd = gr.update(choices=[
                        ("Blog Post", "blog post"),
                    ])
        
        starting_html = """
            <div style="text-align: center; padding: 50px;">
                <h1 style="font-size: 48px; margin-bottom: 20px;">Welcome to Our AI-Assisted Document Creation Task</h1>
                <p style="font-size: 22px; margin-bottom: 40px;">
                    In this task, you will interact with an AI writing assistant to create a <b>blog post</b>. 
                    This task is broken into three simple steps designed to guide you through the experience.
                </p>
                <div style="max-width: 800px; margin: 0 auto; text-align: left;">
                    <ol style="font-size: 20px; line-height: 1.6;">
                        <li>
                            <strong>Step 1: Select Intent that you are interested in</strong><br>
                            Choose an intent that you want to write a blog post about.
                        </li>
                        <li>
                            <strong>Step 2: Pre-Writing Preparation</strong><br>
                            Jot down some pre-writing details for your selected intent, this is for you to organize your thoughts like some rough plan on what to write.
                        </li>
                        <li>
                            <strong>Step 3: Interact with the AI Writing Assistant</strong><br>
                            Talk to the AI to bring your document to life!
                        </li>
                    </ol>
                </div>
            </div>
        """
    elif document_type == "email":
        cw_btn_visible = gr.update(visible=False)
        bp_btn_visible = gr.update(visible=False)
        em_btn_visible = gr.update(visible=True)
        add_own_type_dd = gr.update(choices=[
                        ("Email/Letter", "email"),
                    ])
        starting_html = """
            <div style="text-align: center; padding: 50px;">
                <h1 style="font-size: 48px; margin-bottom: 20px;">Welcome to Our AI-Assisted Document Creation Task</h1>
                <p style="font-size: 22px; margin-bottom: 40px;">
                    In this task, you will interact with an AI writing assistant to create <b>an email</b>. 
                    This task is broken into three simple steps designed to guide you through the experience.
                </p>
                <div style="max-width: 800px; margin: 0 auto; text-align: left;">
                    <ol style="font-size: 20px; line-height: 1.6;">
                        <li>
                            <strong>Step 1: Select Intent that you are interested in</strong><br>
                            Choose an intent that you want to write an email about.
                        </li>
                        <li>
                            <strong>Step 2: Pre-Writing Preparation</strong><br>
                            Jot down some pre-writing details for your selected intent, this is for you to organize your thoughts like some rough plan on what to write.
                        </li>
                        <li>
                            <strong>Step 3: Interact with the AI Writing Assistant</strong><br>
                            Talk to the AI to bring your document to life!
                        </li>
                    </ol>
                </div>
            </div>
        """
    else:
        cw_btn_visible = gr.update(visible=True)
        bp_btn_visible = gr.update(visible=True)
        em_btn_visible = gr.update(visible=True)
        add_own_type_dd = gr.update(choices=[
                        ("Creative Writing", "creative writing"),
                        ("Blog Post", "blog post"),
                        ("Email/Letter", "email"),
                    ])
        select_intent_window_html = """
        <div style="text-align: center; padding: 0;">
            <h2 style="font-size: 32px; margin-bottom: 10px; margin-top: 0">Step 1</h2>
            <p style="font-size: 20px; margin-bottom: 0;">
                Choose one of the following three document types, and then the intent that you are interested in writing about, or add your own intent if you have a specific idea in mind.
            </p>
            <p style="font-size: 20px; color: #D44C4C; margin-top: 0; margin-bottom: 30px;" class="i">
                Note: The final document that you'll create with AI will be between 100 and 500 words, so your intent should not be too simple.
            </p>
        </div>
        """

    user_id_dict = {"user_id": user_id}
    if not state_dict["assignmentId"]:
        return state_dict, user_id_dict, \
                gr.update(visible=True), gr.update(visible=False), cw_btn_visible, bp_btn_visible, em_btn_visible, add_own_type_dd, starting_html, select_intent_window_html
    else:
        if state_dict["assignmentId"] == "ASSIGNMENT_ID_NOT_AVAILABLE":
            return state_dict, user_id_dict, \
                gr.update(visible=False), gr.update(visible=False), cw_btn_visible, bp_btn_visible, em_btn_visible, add_own_type_dd, starting_html, select_intent_window_html
        else:
            return state_dict, user_id_dict, \
                gr.update(visible=False), gr.update(visible=True), cw_btn_visible, bp_btn_visible, em_btn_visible, add_own_type_dd, starting_html, select_intent_window_html
                
def vote(state, data: gr.LikeData):
    if "turn_level_vote" not in state:
        state["turn_level_vote"] = {}
    state["turn_level_vote"][repr(data.index)] = [data.value, data.liked]
    return state

def user(message, history):
    if len(history) > 0 and history[-1]["role"] == "user":
        history = history[:-1]
    return "", history + [{"role": "user", "content": message}], gr.update(visible=True), \
        gr.update(visible=False, elem_classes=["hidden"]), gr.update(visible=True), gr.update(interactive=False, placeholder="Waiting for response...")
    
def echo(history, state):

    history_openai_format = [
        {"role": "system", "content": "You are a skilled writing assistant. Your role is to help users create and edit documents that should be under 600 words by following their specific instructions and requirements."}
    ]
    
    for message in history:
        if message["role"] == "system":
            continue
        history_openai_format.append({"role": message["role"], "content": message["content"]})

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
        
        history_openai_format.append({"role": "assistant", "content": ""})
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                history_openai_format[-1]["content"] += chunk.choices[0].delta.content

                history_openai_format[-1]["content"] = handle_latex_delimeter(history_openai_format[-1]["content"])
                yield history_openai_format
        
    elif state["model"] in ["mistral-large-2407"]:
        # With streaming
        try:
            stream_response = mistral_client.chat.stream(model=state["model"],
                                                          messages=history_openai_format,
                                                          temperature=0,
                                                          max_tokens=2000)
        except Exception as e:
            raise gr.Error(f"An error occurred: {e}")

        history_openai_format.append({"role": "assistant", "content": ""})
        for chunk in stream_response:
            if chunk.data.choices[0].delta.content is not None:

                history_openai_format[-1]["content"] += chunk.data.choices[0].delta.content

                history_openai_format[-1]["content"] = handle_latex_delimeter(history_openai_format[-1]["content"])
                yield history_openai_format

    elif state["model"] in ["claude-3-5-sonnet-20240620"]:
        try:
            with anthropic_client.messages.stream(
                max_tokens=2000,
                messages=history_openai_format[1:], # anthropic does not support system message in messages
                system=history_openai_format[0]["content"],
                model=state["model"],
                temperature=0,
            ) as stream:
                history_openai_format.append({"role": "assistant", "content": ""})
                for text in stream.text_stream:
                    history_openai_format[-1]["content"] += text

                    history_openai_format[-1]["content"] = handle_latex_delimeter(history_openai_format[-1]["content"])
                    yield history_openai_format
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
        
        try:
            response = open_source_client.complete(
                stream=True,
                temperature=0,
                max_tokens=2000,
                messages=history_openai_format
            )

            history_openai_format.append({"role": "assistant", "content": ""})
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    history_openai_format[-1]["content"] += chunk.choices[0].delta.content

                    history_openai_format[-1]["content"] = handle_latex_delimeter(history_openai_format[-1]["content"])
                    yield history_openai_format
                
        except Exception as e:
            raise gr.Error(f"An error occurred: {e}")
    else:
        raise ValueError("Invalid model")


def update_current_document(chatbot, document_history):
    if chatbot[-1]["role"] == "user":
        return len(document_history) - 1, document_history

    user_query = chatbot[-2]["content"]
    model_response = chatbot[-1]["content"]

    if len(document_history) == 0:
        document = ""
    else:
        document = document_history[-1]

    prompt = document_tracking_prompt_template.format(document=document, user_query=user_query, model_response=model_response)
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=10000,
        n=1
    )

    try:
        extracted_json = extract_json(response.choices[0].message.content)
    except Exception as e:
        extracted_json = None

    if extracted_json:
        updated_document = extracted_json["Updated Document"]
        if updated_document == "SAME":
            updated_document = document
        document_history = document_history + [updated_document]
    else:
        document_history = document_history + [document]

    return len(document_history) - 1, document_history

def intent_bank_click(state, intent_bank_clicked_problem, created_documents):
    index = intent_bank_clicked_problem[0]
    intent = intent_bank_clicked_problem[1]
    
    document_type = state["document_type"]

    if [document_type, intent] in created_documents:
        raise gr.Error("You have already created a document with the same intent. Please do another intent.", duration=5)

    bullet_points = intent_bank_data.loc[(intent_bank_data["document_type"] == document_type) & 
                     (intent_bank_data["intent"] == intent), "bullet_points"].values[0]
    

    # Convert JSON string back to dictionary
    bullet_points = list(json.loads(bullet_points).values())

    # Create the ordered list HTML
    bullet_points_html = "\n".join([f"<li class='mv2'>{point}</li>" for point in bullet_points])

    returned_html = f"""
        <div class="ba br3 f4-5 pa3" style="min-height: 450px;">
            <p class="mb3"><span class="b">Selected Intent: </span>{intent}</p>
            <p class="b">Pre-writing Questions for this Intent:</p>
            <ol class="pl4" style="list-style-type: decimal;">
                {bullet_points_html}
            </ol>
            <div class="br3 ph3 pv2 f4-5" style="background-color: #fbe0e0; margin-top: 5pt">
                <p class="mt3">Click the button below to start jotting down some thoughts for the above questions</p>
            </div>
        </div>
    """
    
    state["selected_intent_index"] = index
    state["intent"] = intent

    bullet_points_responses = [""] * len(bullet_points)
    
    return state, bullet_points, bullet_points_responses, bullet_points_responses, returned_html, gr.update(visible=True)

def generate_bullet_points_for_added_intent(state, added_document_type, added_intent, created_documents):
    prompt_template = """You are an expert at generating guiding questions that help writers brainstorm key elements before starting their first draft. Based on the given document type and specific intent, create 10 foundational questions will help writers brainstorm and plan their document.

The questions should:
1. Help establish core elements specific to that document type and intent
2. Guide thinking about audience and purpose

# Input
<Document Type>
{document_type}
</Document Type>
<Intent>
{intent}
</Intent>

# Output Format:
You must output in the following JSON format:
{{
	"Question 1": [Question 1],
	"Question 2": [Question 2],
	"Question 3": [Question 3],
	"Question 4": [Question 4],
	"Question 5": [Question 5],
	"Question 6": [Question 6],
	"Question 7": [Question 7],
	"Question 8": [Question 8],
	"Question 9": [Question 9],
	"Question 10": [Question 10],
}}

# Notes:
1. Questions should be specific to the document type and intent.
2. Focus on essential elements.
3. Frame questions to encourage detailed thinking.
4. Avoid yes/no questions.
5. The document is pure text, so do not include questions related to visual elements. Focus only on textual content."""

    if [added_document_type, added_intent] in created_documents:
        raise gr.Error("You have already created a document with the same intent. Please do another intent.", duration=5)

    if not added_document_type or not added_intent:
        raise gr.Error("Please select a document type and write down your intent before proceeding.", duration=5)

    prompt = prompt_template.format(document_type=added_document_type, intent=added_intent)

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=2000,
        n=1
    )

    response_content = response.choices[0].message.content

    def extract_json(text):
        # Find all potential JSON objects
        match = re.findall(r'\{[^{}]*\}', text)
        if not match:
            return None
        
        last_json_string = match[-1]
        
        # Try to parse the extracted string as JSON
        try:
            return json.loads(last_json_string)
        except json.JSONDecodeError:
            return None

    bullet_points = list(extract_json(response_content).values())

    if not bullet_points:
        raise gr.Error("An error occurred while generating the bullet points. Please try again.", duration=5)

    bullet_points_html = "\n".join([f"<li class='mv2'>{point}</li>" for point in bullet_points])

    returned_html = f"""
        <div class="ba br3 f4-5 pa3" style="min-height: 450px;">
            <p class="mb3"><span class="b">Added Intent: </span>{added_intent}</p>
            <p class="b">Pre-writing Questions for this Intent:</p>
            <ol class="pl4" style="list-style-type: decimal;">
                {bullet_points_html}
            </ol>
            <p class="mt3 i" style="color: #555;">Click the button below to jot down some thoughts for the above questions.</p>
        </div>
    """

    state["intent"] = added_intent
    state["document_type"] = added_document_type

    bullet_points_responses = [""] * len(bullet_points)

    return state, bullet_points, bullet_points_responses, bullet_points_responses, returned_html, gr.update(visible=True)


def select_document_type(state, document_type):

    document_type = document_type.lower()
    print(state)

    if document_type == "email/letter":
        document_type = "email"

    document_type_intent_bank_data = intent_bank_data[intent_bank_data["document_type"] == document_type]

    # reset the index
    document_type_intent_bank_data = document_type_intent_bank_data.reset_index(drop=True)

    # Add the index column
    document_type_intent_bank_data['index'] = document_type_intent_bank_data.index

    cols = ['index', 'intent']

    document_type_intent_bank_data = document_type_intent_bank_data[cols]

    document_type_intent_data_list = document_type_intent_bank_data.values.tolist()
        
    document_type_intent_data_list = [[str(item) for item in row] for row in document_type_intent_data_list]

    state["document_type"] = document_type
    
    return state, gr.Dataset(samples=document_type_intent_data_list), gr.update(visible=True), gr.update(visible=False)


def finish_conversation():
    return gr.update(visible=True), gr.update(visible=True)


def why_stop_other_box_update(why_stop_radio):
    if why_stop_radio == "Other reason":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def submit_click(chatbot, bullet_points_state, bullet_points_responses_state, document_history,
                strength_box, weakness_box, why_stop_radio, why_stop_other_box, 
                overall_interaction_rating, overall_document_rating, feedback_textbox, state,
                focus_loss_state, created_documents):

    user_queries = [turn["content"] for turn in chatbot if turn["role"] == "user"]
    ai_responses = [turn["content"] for turn in chatbot if turn["role"] == "assistant"]
    state["ai_responses"] = ai_responses
    state["user_queries"] = user_queries
    state["document_history"] = document_history
    state["bullet_points"] = bullet_points_state
    state["bullet_points_responses"] = bullet_points_responses_state
    state["strength"] = strength_box
    state["weakness"] = weakness_box
    state["why_stop"] = why_stop_radio
    state["focus_loss_count"] = focus_loss_state
    if why_stop_radio == "Other reason":
        state["why_stop_other_reason"] = why_stop_other_box
    if overall_interaction_rating >= 1 and overall_interaction_rating <= 10:
        state["overall_interaction_rating"] = overall_interaction_rating
    else:
        state["overall_interaction_rating"] = 0
    if overall_document_rating >= 1 and overall_document_rating <= 10:
        state["overall_document_rating"] = overall_document_rating
    else:
        state["overall_document_rating"] = 0
    state["optional_feedback"] = feedback_textbox
    state["end_time"] = time.time()
    state["time_spend"] = state["end_time"] - state["start_time"]
    
    # checking constraints
    if not state.get("cheat", ""):
        if len(user_queries) < 5:
            raise gr.Error("Please chat with the AI writing assistant at least 5 turns as you deeply engage in this document creation before submitting the HIT.", duration=5)
    
        if not state["strength"]:
            raise gr.Error("Please write down the strengths of the AI writing assistant before submitting the HIT.", duration=5)
        if not state["weakness"]:
            raise gr.Error("Please write down the weaknesses of the AI writing assistant before submitting the HIT.", duration=5)
        if not state["why_stop"]:
            raise gr.Error("Please select why you end the session before submitting the HIT.", duration=5)
        if state["why_stop"] == "Other reason" and not state["why_stop_other_reason"]:
            raise gr.Error("Please write down the reason why you end the session before submitting the HIT.", duration=5)
        if not state["overall_interaction_rating"]:
            raise gr.Error("Please rate the overall interaction from 1 to 10 before submitting the HIT.", duration=5)
        if not state["overall_document_rating"]:
            raise gr.Error("Please rate the final document from 1 to 10 before submitting the HIT.", duration=5)
    
    if "document_type" not in state:
        state["document_type"] = "Testing"
        
    if not state["assignmentId"]:
        gr.Info("Thank you for your participation! Your response has been submitted. Jumping to the next one", duration=2)
        
        filename = f"{state['user_id']}_{state['document_type']}_{state['intent'][:20]}"
        filename = sanitize_filename(filename)

        file_path = os.path.join(FOLDER_PATH, state["username"], f"{filename}.json")
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=4)
        
        hf_api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"{state['username']}/{filename}.json",
            repo_id=DATASET_REPO_URL,
            repo_type="dataset",
        )
        time.sleep(2)
    else:
        gr.Info("Thank you for your participation! Your response is being submitted", duration=2)

        filename = f"{state['assignmentId']}_{state['document_type']}_{state['intent'][:20]}"
        filename = sanitize_filename(filename)

        file_path = os.path.join(FOLDER_PATH, state["username"], f"{filename}.json")
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=4)

        hf_api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"{state['username']}/{filename}.json",
            repo_id=DATASET_REPO_URL,
            repo_type="dataset",
        )
        time.sleep(2)

    if state["username"] == "mturk":
        logger.info(f"<><>submit AssignmentId: {state['assignmentId']}, WorkerId: {state['workerId']}, Document Type: {state['document_type']}, has been submitted.")
    else:
        logger.info(f"<><>submit Username: {state['username']}, Document Type: {state['document_type']}, has been submitted.")

    created_documents.append([state["document_type"], state["intent"]])

    return state, created_documents

head = """
    <script>
        document.addEventListener('visibilitychange', function() {
            if (document.hidden) { // Page is no longer visible (focus lost)
                document.getElementById('focus-loss-counter-button').click();
            }
        });
    </script>
    <link rel="stylesheet" href="https://unpkg.com/tachyons@4.12.0/css/tachyons.min.css"/>
"""

with gr.Blocks(delete_cache=(60, 3600),
                fill_height=True,
                fill_width=True,
               title="Create Document with AI",
               head=head,
               css="#chatbot {height: 600px !important;}"
               "#turn-level-rating { height: calc(100vh - 800px) !important; overflow-y: auto !important; overflow-x: hidden !important;}"
               "footer {visibility: hidden;}"
               ".f4-5 {font-size: 1.05rem;}"
               "hr {margin-top: 0.5em; border: none; height: 1.2px; color: #333;  /* old IE */ background-color: #333;  /* Modern Browsers */}"
               ".gap {gap: 6px}"
               "#finish-conversation-button {color: #357EDD}" # blue 
                "#optional-feedback .svelte-1w6vloh .svelte-1w6vloh {font-size: 1.05rem; font-weight: 550; color: #e5a400}" # gold
                "#pre-writing-header .svelte-1w6vloh .svelte-1w6vloh {font-size: 1.05rem; font-weight: 550; color: #73b5f3}"
                "#created-document-accordion {margin-top: 30px}"
                "#created-document-accordion .svelte-1w6vloh .svelte-1w6vloh {font-size: 1.05rem; font-weight: 550;}"
                "#intent-bank .paginate,  #intent-bank .label {font-size: 1.05rem; color: #000000}"
                "#intent-bank .paginate button.svelte-p5q82i {margin-right: 0.5em; margin-left: 0.5em;}"
                "#document-bank-created .paginate,  #document-bank-created .label {font-size: 1.05rem; color: #000000}"
                "#document-bank-created .paginate button.svelte-p5q82i {margin-right: 0.5em; margin-left: 0.5em;}"
                """#add-your-own-button {
                        background-color: #FFEEE6;  /* A soft, natural skin tone */
                        border: none;
                        color: #2B2322;
                        font-weight: 600;
                        }"""
                """.select-document-type-button {
                    font-weight: 600;
                    background-color: #e8e8e8;  /* A soft, natural skin tone */
                    }"""
                """.add-your-own-inputs {
                    padding: 0rem;
                    }
                """
                """#back-to-last-step-button {
                    max-width: 140px !important;
                    # width: 10% !important;
                    background-color: white;
                    border: 2px solid #cccccc;
                    font-weight: 500;
                    padding: 0.2em 0.2em;
                }"""
                """
                #pre_writing_header_questions .svelte-11xb1hd .svelte-phx28p {
                    padding: 0 !important; 
                }"""
                "#control-bar {width: 60%; margin: 0 auto;}"
                "#pre_writing_window {width: 60%; margin: 0 auto;}"
                ".bullet_point {padding:0 !important; margin:0 !important; height:40px; !important}"
                "#canvas_editable {padding:0 !important; margin:0 !important;}"
                ".annotation_box {padding:0 !important; margin:0 !important;}"
                "#msg-box {padding:0 !important; margin:0 !important;}"
                """.new_added_question {padding:0 !important; 
                    margin:0 !important; 
                    height:40px !important;
                    border: 1.5px solid black !important;
                    border-radius: 6px !important;}
                """
                """#remove-bullet-point-button {
                    height: 40px;
                    border-radius: 8px;
                    background-color: #db2727;
                    border: none;
                    color: white;
                    font-size: 15px;
                    line-height: 1;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 0;
                }"""
                """
                #add-bullet-point-button {
                    height: 80px;
                    border-radius: 8px;
                    background-color: #2e7d32;  /* Dark green color */
                    border: none;
                    color: white;
                    font-size: 15px;
                    line-height: 1;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 0;
                }"""
                "#selected-intent {border: 1.5px solid black !important; border-radius: .5rem; padding-top: 0.5rem; padding-bottom: 0.5rem; padding-left: 1rem; padding-right: 1rem;}"
                "#selected-intent span {font-size: 1rem !important;}"
                "#instance-description {border: 1.5px solid black !important; border-radius: .5rem; padding-top: 0.5rem; padding-bottom: 0.5rem; padding-left: 1rem; padding-right: 1rem;}"
                "#instance-description span {font-size: 1rem !important;}"
                ".centered-column { width: 100% !important; margin-left: auto !important; margin-right: auto !important;}"
                ".scrollable-row {max-height:2000px !important; overflow-y: auto;}"
                ".scrollable-row > .gr-column {flex: 1; display: flex;flex-direction: column;}"
                "#canvas_column { height: 630px; /* Full viewport height */ display: flex; flex-direction: column; scroll-behavior: smooth; overflow-y: auto;}"
                "#canvas_intro {display: flex; align-items: center; justify-content: center; height: 100%;}"
                "#canvas_intro .intro_container {text-align: center; }"
                "#canvas_markdown {text-align: left; padding-top:5%; padding-bottom:5%; padding-left: 10%; padding-right: 10%;}"
                "#canvas_markdown .md {font-size: 15px; !important}"
                "#canvas_intro.hidden {display: none; /* Remove from layout when hidden */}"
                """
                    .canvas_toolbar_button {
                        background-color: white;
                        font-weight: 500;
                        border: 2px solid #d9d9d9;
                        max-width: 70px !important;
                        min-width: 70px !important;
                        padding: 0.2em 0.2em;
                    }
                """
                "#domain_specific_d_type_row {width: 80% !important}"
                "#other_d_type_row {width: 20% !important}"
            ) as demo:
    
    user_id_dict = gr.JSON(visible=False)
    created_documents = persist(gr.State(value=[]), cookies)
    state = gr.JSON(visible=False)
    bullet_points_state = gr.State([])
    bullet_points_responses_state = gr.State([])
    bullet_points_responses_state_temp = gr.State([])
    document_history = gr.State([])
    current_document_index = gr.State(-1)
    step_index = gr.State(0)
    focus_loss_state = gr.State(0)

    back_to_last_step_button = gr.Button(value="Back to Last Step", elem_classes=["mt-4", "dim"], elem_id="back-to-last-step-button", visible=True, interactive=False)

    step_index.change(
        fn=lambda step_index: gr.update(interactive=False) if step_index == 0 else gr.update(interactive=True),
        inputs=[step_index],
        outputs=[back_to_last_step_button]
    )

    focus_loss_counter_button = gr.Button(visible=False, elem_id="focus-loss-counter-button")  # Hidden button

    focus_loss_counter_button.click(
        lambda x: x+1,
        inputs=[focus_loss_state],
        outputs=[focus_loss_state]
    )

    
    with gr.Column(elem_classes=["mt2"], visible=True) as starting_window:
        starting_html = gr.HTML()

        def check_both_boxes(checkbox1, checkbox2):
            """
            Return interactive=True only if both checkboxes are checked.
            """
            if checkbox1 and checkbox2:
                return gr.update(interactive=True)
            else:
                return gr.update(interactive=False)

        with gr.Row():
            # Column for original data collection notice
            with gr.Column(elem_classes=["ba", "pa3"]):
                gr.HTML("""
                    <div class="f4-5">
                    <p><b>Data Collection Notice:</b></p>
                    <p>Before you begin, please note that by checking the box below, you agree to:
                    <ul>
                        <li>Allow us to collect your annotations for research</li>
                        <li>Have your annotations shared publicly as part of our research data</li>
                    </ul>
                    So you should not include any personal identifying information (PII) in your annotations.
                    If you prefer not to participate, you can simply close this window.
                    </p>
                    </div>
                """)
                agreement_check_box1 = gr.Checkbox(
                    label="I agree that my annotations will be collected and shared publicly as research data.",
                    value=False,
                    elem_id="consent-checkbox",
                    interactive=True,
                    elem_classes=["f4-5"]
                )

            # Column for the new note
            with gr.Column(elem_classes=["ba", "pa3"]):
                gr.HTML("""
                    <p style="margin-bottom: 1 !important; margin-top: 0 !important; background-color: #f0f0f0; color: #333; padding: 10px; border-radius: 5px;" class="f4-5">
                        <b>Note:</b> 
                        Please view this page in light mode for the best experience, as dark mode may make the text less visible.
                    </p>
                    <p style="margin-bottom: 0 !important; margin-top: 0 !important;" class="f4-5">
                        <b>Note (Dec 30th):</b> 
                        You should try to create content that's ready for immediate use in your real life.
                        For example, when writing emails, instead of just filling the information, 
                        you can also ask AI to customize them to match your personal writing style. Thanks!
                    </p>
                    <p style="margin-bottom: 0 !important; margin-top: 0 !important;" class="f4-5">
                        <b style="color: #4169E1">New Note (Jan 3rd):</b> 
                        Please don’t paste all your pre-writing notes at once in one turn when you chatting with the AI. 
                        Think of these as your private brainstorming steps—share them gradually, like how you naturally talk 
                        to a chatbot and develop a document.
                    </p>
                """)
                agreement_check_box2 = gr.Checkbox(
                    label="I have read these notes carefully.",
                    value=False,
                    interactive=True,
                    elem_classes=["f4-5"]
                )

        # Add a button at the bottom
        starting_button = gr.Button(
            value="Let's Start the Task",
            variant="primary",
            size="lg",
            elem_classes=["mt-4"],  # Adds spacing above the button
            interactive=False
        )

        # Use the same function on the change events of both checkboxes
        # The function will check if both are True, then update the button accordingly.
        agreement_check_box1.change(
            fn=check_both_boxes,
            inputs=[agreement_check_box1, agreement_check_box2],
            outputs=starting_button
        )
        agreement_check_box2.change(
            fn=check_both_boxes,
            inputs=[agreement_check_box1, agreement_check_box2],
            outputs=starting_button
        )

    with gr.Column(elem_classes=["mt2"], visible=False) as select_intent_window:
        # Add HTML section for step title
        select_intent_window_html = gr.HTML()

        with gr.Row() as general_purpose_document_types:
            creative_writing_button = gr.Button(value="Creative Writing", scale=1, elem_classes=["dim", "select-document-type-button"], visible=True)
            blog_post_button = gr.Button(value="Blog Post", scale=1, elem_classes=["dim", "select-document-type-button"], visible=False)
            email_button = gr.Button(value="Email/Letter", scale=1, elem_classes=["select-document-type-button", "dim"], visible=False)
            add_your_own_button = gr.Button(value="Work on Your Own Intent", scale=1, 
                                            elem_id="add-your-own-button", elem_classes=["dim"])
        
        with gr.Row(visible=False) as bank_and_selection_row:
            intent_bank = gr.Dataset(components=["markdown", "markdown"], headers=["Index", "Intent"],
                                        label="Intent Bank", elem_id="intent-bank", samples_per_page=10, scale=1)

            with gr.Column(scale=1, elem_classes=["mt4"]) as confirmation_col:
                selected_intent_information_html = gr.HTML(f"""
                    <div class="ba br3" style="display: flex; min-height: 450px;
                                        justify-content: center; align-items: center;">
                        <span class="f4-5">Click to select your interested intent from the intent bank</span>
                    </div>""")
                
                move_to_jotting_button = gr.Button(value="Proceed to Step 2: Pre-Writing Preparation", 
                                                   elem_id="confirm-problem-button",
                                                    variant="primary",
                                                    size="lg",
                                                   visible=False)
                
        with gr.Row(visible=False) as add_your_own_row:
            with gr.Column(scale=1,) as workspace_col:
                gr.HTML("""<p style="font-size: 1.05rem;">🔨 Workspace</p>
                        <p class="b f4-5">What's your document type? [Choose one from Creativing Writing, Blog Post, or Email/Letter]</p>""")
                
                add_your_own_row_dd = gr.Dropdown(choices=[
                        ("Creative Writing", "creative writing"),
                        ("Blog Post", "blog post"),
                        ("Email/Letter", "email"),
                    ], show_label=False, 
                    elem_classes=["add-your-own-inputs"], 
                    visible=True,
                    interactive=True)

                gr.HTML("""
                    <p class="b f4-5">What's your intent?</p>
                    <p class="f6 gray mb3">Here are some examples to help you choose. Please format your input as "[Intent Name]: [Intent Description]"</p>
                    <ul class="list pl0 ml0 mw6 ba b--light-silver br2 pa3 bg-washed-yellow mb3">
                        <li class="ph3 pv2 bb b--light-silver">
                            <span class="f6 fw6 dark-gray">Example 1:</span>
                            <p class="f6 gray mt1 mb0">Epic Fantasy: Write a story about a hero's journey in an imagined world filled with mythical creatures, fantastical realms, and challenges.</p>
                        </li>
                        <li class="ph3 pv2 bb b--light-silver">
                            <span class="f6 fw6 dark-gray">Example 2:</span>
                            <p class="f6 gray mt1 mb0">Travel Guide: Share details of your long trip or travel, and provide advice for future travelers.</p>
                        </li>
                        <li class="ph3 pv2">
                            <span class="f6 fw6 dark-gray">Example 3:</span>
                            <p class="f6 gray mt1 mb0">Job Application: Write a cover letter expressing interest in a position, highlighting relevant skills and experiences.</p>
                        </li>
                    </ul>
                    """)

                add_your_own_row_intent = gr.Textbox(lines=2, show_label=False, visible=True, 
                           elem_classes=["add-your-own-inputs"], interactive=True, placeholder='Please write in "[Intent Name]: [Intent Description]" format')
                
                run_bullet_points_button = gr.Button(value="Generate pre-writing questions for your intent",
                                                        elem_id="run-bullet-points-button",
                                                        variant="primary",
                                                        size="lg",
                                                        visible=True)
            
                

            with gr.Column(scale=1, elem_classes=["mt4"]) as add_your_own_confirmation_col:
                add_your_own_confirmation_html = gr.HTML(f"""
                    <div class="ba br3" style="display: flex; min-height: 450px;
                                        justify-content: center; align-items: center;">
                        <span class="f4-5">Your pre-writing questions will appear here.</span>
                    </div>""")
                
                move_to_jotting_button_add_own = gr.Button(value="Proceed to Step 2: Pre-Writing Preparation", 
                                                   elem_id="confirm-problem-button",
                                                    variant="primary",
                                                    size="lg",
                                                   visible=False)


        with gr.Accordion("Documents that you have created with AI", 
                          elem_id="created-document-accordion") as created_document_accordion:
            created_documents_data =  gr.Dataset(components=["markdown", "markdown"],
                                                headers=["Document Type", "Intent"],
                                                label="Created Documents" ,samples_per_page=5, elem_id="document-bank-created")
            
        add_your_own_button.click(
            lambda: (gr.update(visible=False), gr.update(visible=True)),
            inputs=None,
            outputs=[bank_and_selection_row, add_your_own_row]
        )    
                                  
        intent_bank.click(
            fn=intent_bank_click,
            inputs=[state, intent_bank, created_documents],
            outputs=[state, bullet_points_state, bullet_points_responses_state, 
                     bullet_points_responses_state_temp, selected_intent_information_html, move_to_jotting_button]
        )

        run_bullet_points_button.click(
            fn=generate_bullet_points_for_added_intent,
            inputs=[state, add_your_own_row_dd, add_your_own_row_intent, created_documents],
            outputs=[state, bullet_points_state, bullet_points_responses_state, 
                     bullet_points_responses_state_temp, add_your_own_confirmation_html, move_to_jotting_button_add_own]
        )
        

        creative_writing_button.click(
            fn=select_document_type,
            inputs=[state, creative_writing_button],
            outputs=[state, intent_bank, bank_and_selection_row, add_your_own_row]
        )
        blog_post_button.click(
            fn=select_document_type,
            inputs=[state, blog_post_button],
            outputs=[state, intent_bank, bank_and_selection_row, add_your_own_row]
        )
        email_button.click(
            fn=select_document_type,
            inputs=[state, email_button],
            outputs=[state, intent_bank, bank_and_selection_row, add_your_own_row]
        )

    starting_button.click(
            lambda: (gr.update(visible=False), gr.update(visible=True), 1),
            inputs=None,
            outputs=[starting_window, select_intent_window, step_index]
        )
    
    with gr.Column(visible=False, elem_id="pre_writing_window") as pre_writing_window:
        gr.HTML("""
        <div style="text-align: center; padding: 0;">
            <h2 style="font-size: 32px; margin-bottom: 10px; margin-top: 0">Step 2</h2>
            <p style="font-size: 20px; margin-bottom: 30px;">
                Pre-writing: Jot down some thoughts for your intent.
            </p>
        </div>
        """)

        selected_intent = gr.HTML()

        with gr.Row(visible=True, elem_classes=["mt2"]):
            with gr.Column(scale=6, elem_id="pre_writing_header_questions"):
                gr.HTML("""
                    <div class="bg-light-gray pa3 br3 lh-copy f4-5" style="min-height:83px; background-color: #fbe0e0">
                        Note: You can customize the questions to suit your needs—feel free to add new ones, remove existing ones, or leave fields blank if certain aspects don’t apply to your planning. However, please make sure 
                        that you answer at least <b>6 questions</b>.
                    </div>
                    <div class="f4-5 mt2">
                        <b style="color: #4169E1">Bug Note:</b> 
                        There’s a bug where sometimes adding a new question and clicking “Save” doesn't actually save. A workaround is to add several new questions at once, save ones that you need, and then remove any that you don’t need.
                """)
            add_new_bullet_point_button = gr.Button(value="Add New Question", 
                                                scale=1, elem_classes=["dim"],
                                                elem_id="add-bullet-point-button")
            
        add_new_bullet_point_button.click(
            fn=lambda points, responses: (points + [""], responses + [""], responses + [""]),
            inputs=[bullet_points_state, bullet_points_responses_state],
            outputs=[bullet_points_state, bullet_points_responses_state, bullet_points_responses_state_temp]
        )
        
        # update bullet_points_responses_state_temp except box change

        @gr.render(inputs=[bullet_points_state, bullet_points_responses_state_temp])
        def list_bullet_point_with_text_box(bullet_points, bullet_points_responses_temp):
            for i, bullet_point in enumerate(bullet_points):
                
                if bullet_point:
                    gr.HTML(f"<p class='f5' style='margin-bottom: 5px;'>{i + 1}. {bullet_point}</p>")  # Smaller font for the bullet point
                else:
                    with gr.Row():
                        new_added_question_textbox = gr.Textbox(show_label=False, visible=True, interactive=True,
                                                            elem_classes=["new_added_question"], 
                                                            placeholder='Enter your question here and Click "Save" button to save', scale=8)
                        new_added_question_confirm_button = gr.Button(value="Save",
                                                                    elem_classes=["dim"], 
                                                                    interactive=True, scale=1)
                    
                        def update_bullet_point(bullet_points, bullet_points_responses_state, value, index=i):
                            bullet_points[index] = value
                            return bullet_points, bullet_points_responses_state

                        new_added_question_confirm_button.click(
                            fn=update_bullet_point,
                            inputs=[bullet_points_state, bullet_points_responses_state, new_added_question_textbox],
                            outputs=[bullet_points_state, bullet_points_responses_state_temp]
                        )

                with gr.Row():
                    box = gr.Textbox(bullet_points_responses_temp[i], show_label=False, visible=True, elem_classes=["bullet_point"], scale=8)  # Add custom class
                    remove_bullet_point_button = gr.Button(value="Remove",elem_classes=["dim"], 
                                                            elem_id="remove-bullet-point-button")
                    
                    def remove_bullet_point(bullet_points, bullet_points_responses, index=i):
                        if len(bullet_points) == 6:
                            raise gr.Error("You must have at least 6 questions to proceed.", duration=5)
                        bullet_points.pop(index)
                        bullet_points_responses.pop(index)
                        return bullet_points, bullet_points_responses, bullet_points_responses
                    
                    remove_bullet_point_button.click(
                        fn=remove_bullet_point,
                        inputs=[bullet_points_state, bullet_points_responses_state],
                        outputs=[bullet_points_state, bullet_points_responses_state, bullet_points_responses_state_temp]
                    )

                    def update_bullet_point_responses_state(bullet_points_responses_state, value, index=i):
                        bullet_points_responses_state[index] = value
                        return bullet_points_responses_state

                    box.change(
                        fn=update_bullet_point_responses_state,
                        inputs=[bullet_points_responses_state, box],
                        outputs=[bullet_points_responses_state]
                    )
            

        move_to_conversation_button = gr.Button(value="Proceed to Step 3: Converse with the AI Assistant",
                                                variant="primary",
                                                size="lg")
        

    def move_to_jotting_click(state):
        intent = state["intent"]
        selected_intent_html = f"""
            <div class="ba br3 f4-5 pa3 bw1">
                <p class="mb3"><span class="b">Your Intent: </span>{intent}</p>
            </div>
        """
                
        return gr.update(visible=False), gr.update(visible=True), selected_intent_html, 2

    gr.on(
        triggers=[move_to_jotting_button.click, move_to_jotting_button_add_own.click],
        fn=move_to_jotting_click,
        inputs=state,
        outputs=[select_intent_window, pre_writing_window, selected_intent, step_index]
    )
    
    with gr.Column(visible=False) as conversation_window:
        gr.HTML("""
            <div style="text-align: center; padding: 0;">
                <h2 style="font-size: 32px; margin-bottom: 10px; margin-top: 0">Step 3</h2>
                <p style="font-size: 20px; margin-bottom: 30px;">
                    Converse with the AI assistant to create your document.
                </p>
            </div>
            """)


        with gr.Accordion("You can check your pre-writing responses here", open=False, elem_id="pre-writing-header"):
            with gr.Column(scale=2, elem_classes=["ba pa2 bw1 b--black-60 br3 centered-column"]):
                
                @gr.render(inputs=[bullet_points_state, bullet_points_responses_state])
                def list_bullet_point_with_response(bullet_points, bullet_points_responses):
                    with gr.Row():
                        # Left column (odd numbers: 1, 3, 5, 7, 9)
                        with gr.Column(scale=1):
                            for i in range(0, len(bullet_points), 2):  # Step by 2 to get odd indices
                                gr.HTML(f"<p class='f5' style='margin-bottom: 5px;'>{i + 1}. {bullet_points[i]}</p>")
                                box = gr.Textbox(show_label=False, visible=True, 
                                            elem_classes=["bullet_point"], interactive=False,
                                            value=bullet_points_responses[i])
                        
                        # Right column (even numbers: 2, 4, 6, 8, 10)
                        with gr.Column(scale=1):
                            for i in range(1, len(bullet_points), 2):  # Step by 2 starting from 1 to get even indices
                                gr.HTML(f"<p class='f5' style='margin-bottom: 5px;'>{i + 1}. {bullet_points[i]}</p>")
                                gr.Textbox(show_label=False, visible=True, 
                                            elem_classes=["bullet_point"], interactive=False,
                                            value=bullet_points_responses[i])

        gr.HTML(f"""<div class="br3 ph3 pv2 f4-5" style="background-color: #fbe0e0; margin-top: 5pt">
            <p><b>Please follow these guidelines to ensure meaningful and productive interactions with the AI writing assistant:</b></p>
            <ul>
                <li>Have at least 5 meaningful exchanges with the AI, making a genuine effort to engage thoughtfully and avoiding quick or shallow interactions.</li>
                <li>Besides writing the document, feel free to ask the AI to brainstorm ideas or provide knowledge to improve the content.</li>
                <li>Make sure the final document is between 100 and 500 words to include enough detail and clarity.</li>
            </ul>
            <p style="margin-bottom: 0 !important"><b>Note:</b> 1. Your pre-writing responses are for organizing your own thoughts - the AI writing assistant won't have access to them. Use them as your personal 
                reference while talking with the AI, and don't worry if your discussion with the AI takes a different direction 
                than what you initially planned. </p>
            <p class="mt0 mb0">2. Give a 👍 if you find the AI's response quite helpful, or a 👎 if you think it is bad.</p>
            <p class="mt0 mb0">
                3. Create content that's ready for immediate use in your real-world context. 
                For example, when writing emails, instead of just filling the information, you can also ask AI to customize them to match your personal writing style.
            </p>
            <p style="margin-bottom: 0 !important; margin-top: 0 !important;"><b style="color: #4169E1">New Note (Jan 3rd):</b>
                Please don’t paste all your pre-writing notes at once in one turn. 
                Think of these as your private brainstorming steps—share them gradually, like how you naturally talk 
                to a chatbot and develop a document.
            </p>
               
            
        </div>""")

        with gr.Row(elem_classes=["mt2 scrollable-row"], visible=True):

            # ba pa2 bw1 b--black-60 br3 b--blue 
            with gr.Column(scale=4):
                with gr.Column(scale=3, elem_id="col"):
                    chatbot = gr.Chatbot(
                            type="messages",
                            label="AI Writing Assistant",
                            render=True,
                            elem_id="chatbot",
                            avatar_images=(None, "img/ai.png"),
                            latex_delimiters=latex_delimeter_set
                        )
                    msg = gr.Textbox(show_label=False, placeholder="Chat with the AI writing assistant...", interactive=True, elem_id="msg-box")
                    chatbot.like(vote, inputs=state, outputs=state)
                        
            with gr.Column(scale=6, elem_id="canvas_column") as canvas:
                
                canvas_intro = gr.HTML("""<div class="intro_container">
                                    <p class="f3">Hello!</p>
                                    <p class="f3">The document will be shown here when you engage with the AI writing assistant.</p>
                                </div>""", elem_id="canvas_intro", visible=True)
                with gr.Column(visible=True) as canvas_main:
                    with gr.Row() as canvas_toolbar:
                        with gr.Column(scale=6):
                            gr.HTML("""
                                <div class="bg-light-gray ph2 pv1 br3 f6 gray lh-copy f4-5">
                                    These buttons are for fixing rare merge errors only. Please do not edit the document unless you are correcting a merge issue. This canvas is for your viewing-only, not for editing.
                                </div> 
                            """)
                        back_button = gr.Button(value="Previous", elem_id="back-button", elem_classes=["canvas_toolbar_button", "dim"], interactive=False)
                        forward_button = gr.Button(value="Current", elem_id="forward-button", elem_classes=["canvas_toolbar_button", "dim"], interactive=False)
                        edit_button = gr.Button(value="Edit", elem_id="edit-button", elem_classes=["canvas_toolbar_button", "dim"], interactive=False)
                        save_button = gr.Button(value="Save", elem_id="save-button", interactive=False, elem_classes=["canvas_toolbar_button", "dim"])

                    processing_tip = gr.HTML("""<div class="ph2 pv1 br3 f6 lh-copy f4-5" style="color: #FF725C">
                        (1/2) The AI writing assistant is generating response to your query. Please wait for a moment...
                    </div>""", visible=False)
                    merging_tip = gr.HTML("""<div class="ph2 pv1 br3 f6 lh-copy f4-5" style="color: #E7040F">
                        (2/2) We are merging AI's response with the document. Please wait for a moment...
                    </div>""", visible=False)
                    canvas_markdown = gr.Markdown(latex_delimiters=latex_delimeter_set, elem_id="canvas_markdown", visible=True)
                    canvas_editable = gr.Textbox(lines=50, label=None, show_label=False, 
                                           placeholder="Write down your step-by-step solution here.", interactive=True, visible=False,
                                           elem_id="canvas_editable")
                
                msg.submit(user, [msg, chatbot], [msg, chatbot, canvas_main, canvas_intro, processing_tip, msg], queue=False).then(echo, [chatbot, state], [chatbot]).then(
                    lambda: gr.update(visible=True), None, [merging_tip]
                ).then(
                        update_current_document, [chatbot, document_history], [current_document_index, document_history]
                ).then(
                        lambda : (gr.update(interactive=True, placeholder="Chat with the AI writing assistant..."),
                            gr.update(visible=False), gr.update(visible=False)), None, [msg, processing_tip, merging_tip,])

                back_button.click(
                    fn=lambda x: len(x) - 2,
                    inputs=[document_history],
                    outputs=[current_document_index]
                )

                forward_button.click(
                    fn=lambda x: len(x) - 1,
                    inputs=[document_history],
                    outputs=[current_document_index]
                )

                def edit_button_click(current_document_index, document_history):
                    return gr.update(value=document_history[current_document_index], visible=True), gr.update(visible=False), gr.update(interactive=False), gr.update(interactive=True)
                

                edit_button.click(
                    fn=edit_button_click,
                    inputs=[current_document_index, document_history],
                    outputs=[canvas_editable, canvas_markdown, edit_button, save_button]
                )

                def save_button_click(state, document_history, canvas_editable):
                    state["edit_document"] = True
                    state["edit_document_index"] = len(document_history) - 1
                    state["edit_document_old"] = document_history[-1]
                    document_history[-1] = canvas_editable
                    return state, document_history, gr.update(value=canvas_editable, visible=True), gr.update(visible=False), gr.update(interactive=True), gr.update(interactive=False)
                

                save_button.click(
                    fn=save_button_click,
                    inputs=[state, document_history, canvas_editable],
                    outputs=[state, document_history, canvas_markdown, canvas_editable, edit_button, save_button]
                )

                def current_document_change(current_document_index, document_history):
                    if current_document_index == len(document_history) - 1:
                        return document_history[current_document_index], gr.update(interactive=True), gr.update(interactive=False), \
                                gr.update(interactive=True)
                    elif current_document_index == len(document_history) - 2:
                        if current_document_index == -1:
                            current_document = ""
                        else:
                            current_document = document_history[current_document_index]
                        return current_document, gr.update(interactive=False), gr.update(interactive=True), \
                            gr.update(interactive=True)

                
                current_document_index.change(
                    fn=current_document_change,
                    inputs=[current_document_index, document_history],
                    outputs=[canvas_markdown, back_button, forward_button, edit_button]
                )

        
        finish_conversation_instruction = gr.HTML(f"""<div class="br3 ph3 pv2 f4-5 mt3" style="background-color: #cceeff">
                            Click <span class="i" style="color: #357EDD">Finish Conversation</span> when you are satisfied with the document or if you feel the AI writing assistant is no longer helpful. You will then be asked to answer <b>five survey questions</b> about your experience with the AI writing assistant.
                        </div>""", visible=True)
        finish_conversation_button = gr.Button(value="Finish Conversation", elem_id="finish-conversation-button", visible=True)

        with gr.Column(visible=False) as annotation_column:
            gr.HTML("""
                    <p class="f4-5 b">Five survey questions:</p>
                    """)
            
            with gr.Row():
                with gr.Column():
                    gr.HTML("""
                        <p class="f4-5">1. What are the strengths of the AI writing assistant?</p>
                        """)
                    strength_box = gr.Textbox(show_label=False, elem_classes=["annotation_box"], visible=True, placeholder="strengths of the AI writing assistant...")
                with gr.Column():
                    gr.HTML("""
                        <p class="f4-5">2. What are the weaknesses of the AI writing assistant?</p>
                        """)
                    weakness_box = gr.Textbox(show_label=False, elem_classes=["annotation_box"], visible=True, placeholder="weaknesses of the AI writing assistant...")

            gr.HTML("""
                <p class="f4-5">3. Why did you end the session?</p>
                """)
            why_stop_radio = gr.Radio(["I am satified with the document the AI created.", "I found the AI writing assistant not helpful.", "I encountered technical issues with the interface.", "Other reason"], show_label=False, visible=True)
            why_stop_other_box = gr.Textbox(lines=1, label="Write down your reason", visible=False)

            with gr.Row(elem_classes=["pa1"]):
                with gr.Column(scale=3):
                    gr.HTML(
                        """ 
                        <p class="f4-5">5. Please rate your <span style="color: red !important; font-weight: 600">interaction experience</span> (the conversation) with the AI writing assistant from 1 to 10 based on the following criteria:</p>
                        <ul class="f4-5 list">
                            <li>Score 1 ~ 2 <span class="b"> (very poor)</span>: The writing assistant consistently failed to understand your inputs, provided irrelevant or nonsensical responses, and made the interaction frustrating and unproductive.</li>
                            <li>Score 3 ~ 4 <span class="b"> (poor)</span>: The writing assistant frequently misunderstood your requests, offered minimal assistance that didn't address your needs, and required repeated clarifications.</li>
                            <li>Score 5 ~ 6 <span class="b"> (average)</span>: The writing assistant was somewhat helpful but had noticeable issues with comprehension or responsiveness, providing partially relevant responses that contained errors or omissions.</li>
                            <li>Score 7 ~ 8 <span class="b"> (good)</span>: The writing assistant generally understood your inputs and provided relevant, useful responses that effectively aided your document creation, with only minor issues.</li>
                            <li>Score 9 ~ 10 <span class="b"> (very good)</span>: The writing assistant consistently demonstrated clear understanding and provided insightful, comprehensive assistance that exceeded expectations and significantly enhanced your efficiency and outcomes.</li>
                        </ul>
                        """
                    )
                with gr.Column(scale=1):
                    overall_interaction_rating = gr.Slider(value=0, label="Interaction Rating", minimum=1, maximum=10, step=1, 
                                            info="rate from 1 (worst) to 10 (best)",interactive=True)

            with gr.Row(elem_classes=["pa1"]):
                with gr.Column(scale=3):
                    gr.HTML(
                        """ 
                        <p class="f4-5">5. Please rate the <span style="color: green !important; font-weight: 600">final document</span> created by the AI writing assistant from 1 to 10 based on the following criteria:</p>
                        <ul class="f4-5 list">
                            <li>Score 1 ~ 2 <span class="b"> (very poor)</span>: The document contains numerous errors, inaccuracies, or irrelevant content, lacks coherence and structure, and is unusable for your objective. </li>
                            <li>Score 3 ~ 4 <span class="b"> (poor)</span>: The document has significant issues such as incomplete sections, misleading information, or poor organization, only partially addresses your instructions, and requires substantial revisions. </li>
                            <li>Score 5 ~ 6 <span class="b"> (average)</span>: The document meets basic requirements but includes noticeable errors or omissions, provides some useful content but lacks depth or clarity, and requires moderate revisions to improve quality. </li>
                            <li>Score 7 ~ 8 <span class="b"> (good)</span>: The document is well-organized, covers the key topics as instructed, contains accurate and relevant information with minor errors, and serves as a strong foundation that fulfills your main objective. </li>
                            <li>Score 9 ~ 10 <span class="b"> (very good)</span>: The document is comprehensive, insightful, and meticulously crafted, exceeds expectations by providing exceptional clarity and depth, requires minimal to no revisions, and significantly achieves your objective.</li>
                        </ul>
                        """
                    )
                with gr.Column(scale=1):
                    overall_document_rating = gr.Slider(value=0, label="Document Rating", minimum=1, maximum=10, step=1, 
                                            info="rate from 1 (worst) to 10 (best)",interactive=True)

        with gr.Column(visible=False) as final_button_col:
                with gr.Row():
                    submit_hit_button = gr.Button(value="Submit the Hit", visible=False, interactive=True)
                    submit_hf_button = gr.Button(value="Submit the Task", visible=False, interactive=True)

        with gr.Accordion("Optional feedback on the interface or the task", open=False, elem_id="optional-feedback"):
            with gr.Column():
                feedback_textbox = gr.Textbox(lines=5, label="""Please leave any feedback or comments you have about the interface or the task here. Let us know how we can improve.""", visible=True)


    def move_to_conversation_button_click(state, bullet_points, bullet_points_responses):
        if not state.get("cheat", ""):
            valid_responses = sum(1 for response in bullet_points_responses if len(response.split()) >= 2)
        
            # Check if we have at least 6 valid responses
            if valid_responses < 6:
                raise gr.Error("Please answer at least 6 questions to proceed.", duration=5)
            
            # check if there is empty question in the bullet points
            if "" in bullet_points:
                raise gr.Error("You have some questions that haven't been saved, please click 'Enter' or 'Return' to save them", duration=5)
        
        return gr.update(visible=False), gr.update(visible=True), 3

    move_to_conversation_button.click(
        fn=move_to_conversation_button_click,
        inputs=[state, bullet_points_state, bullet_points_responses_state],
        outputs=[pre_writing_window, conversation_window, step_index]
    )
    
    output = gr.Textbox(lines=2, label="AI Response", visible=False)


    why_stop_radio.select(why_stop_other_box_update, inputs=[why_stop_radio], outputs=[why_stop_other_box])


    finish_conversation_button.click(finish_conversation, outputs=[annotation_column, final_button_col])

    def back_to_last_step_button_click(step_index):
        if step_index == 1:
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), step_index-1
        elif step_index == 2:
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), step_index-1
        elif step_index == 3:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), step_index-1

    back_to_last_step_button.click(
        fn=back_to_last_step_button_click,
        inputs=[step_index],
        outputs=[starting_window, select_intent_window, pre_writing_window, conversation_window, step_index]
    )

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
    
    ##  submit to huggingface
    submit_hf_button.click(submit_click, inputs=[chatbot,
                                bullet_points_state, bullet_points_responses_state, document_history,
                                strength_box, weakness_box, why_stop_radio, why_stop_other_box, overall_interaction_rating, 
                                overall_document_rating, feedback_textbox, state, focus_loss_state, created_documents],
                                outputs=[state, created_documents]).success(lambda state: state, inputs=[state], outputs=[state], js=refresh_webpage_js)
    
    
    ## submit to mturk
    submit_hit_button.click(submit_click, inputs=[chatbot,
                                bullet_points_state, bullet_points_responses_state, document_history,
                                strength_box, weakness_box, why_stop_radio, why_stop_other_box, overall_interaction_rating, 
                                overall_document_rating, feedback_textbox, state, focus_loss_state, created_documents],
                                outputs=[state, created_documents]).success(lambda state: state, inputs=[state], outputs=[state], js=post_hit_js)
    
    
    cookie_js = '''
        function(value){
            let user_id = value['user_id']; // Access the user_id from the value dictionary
            document.cookie = 'user_id=' + user_id + '; Path=/;  SameSite=None; Secure'; // this allows iframe like in amt
            return value;
        }
    '''

    def update_and_print_created_documents(created_documents):
        if created_documents:
            return gr.Dataset(samples=created_documents), gr.update(visible=True)
        return gr.Dataset(samples=None), gr.update(visible=False)

    demo.load(load_instance, None, outputs=[state, user_id_dict,
                                            submit_hf_button, 
                                            submit_hit_button,
                                            creative_writing_button, blog_post_button, email_button, add_your_own_row_dd, starting_html, select_intent_window_html]).then(
                            lambda user_id: None, inputs=[user_id_dict], js=cookie_js).then(
                                update_and_print_created_documents, inputs=[created_documents], 
                                outputs=[created_documents_data, created_document_accordion]
                            )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=8, max_size=8)
    demo.launch(max_threads=50)
