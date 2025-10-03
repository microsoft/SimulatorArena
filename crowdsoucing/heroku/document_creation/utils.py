import os
from promptflow.core import tool

from gradio.context import Context
from gradio import Request
import gradio as gr
import re
import json

from dotenv import load_dotenv
dotenv_path = os.path.expanduser('~/.env')
load_dotenv(dotenv_path)


def rreplace(s, old, new, occurrence):
  li = s.rsplit(old, occurrence)
  return new.join(li)

def sanitize_filename(value: str) -> str:
  """
  Replace special characters in a string with a hyphen (-) to make it safe for file paths.

  Args:
      value (str): The string to sanitize.

  Returns:
      str: The sanitized string.
  """
  # Replace any character that is not alphanumeric, underscore, or hyphen with a hyphen
  sanitized = re.sub(r'[^\w\-]', '-', value)
  # Remove consecutive hyphens
  sanitized = re.sub(r'-{2,}', '-', sanitized)
  # Trim hyphens from start and end
  return sanitized.strip('-')
      
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

def handle_latex_delimeter(partial_message):
  if partial_message.count("\\[") > partial_message.count("\\\\["):
    partial_message = rreplace(partial_message, "\\[", "\\\\[", 1)
  if partial_message.count("\\]") > partial_message.count("\\\\]"):
    partial_message = rreplace(partial_message, "\\]", "\\\\]", 1)
  if partial_message.count("\\(") > partial_message.count("\\\\("):
    partial_message = rreplace(partial_message, "\\(", "\\\\(", 1)
  if partial_message.count("\\)") > partial_message.count("\\\\)"):
    partial_message = rreplace(partial_message, "\\)", "\\\\)", 1)
  return partial_message

def persist(component, cookies):
    sessions = cookies

    def resume_session(value, request: Request):
      return sessions.get(request.cookies.get('user_id', ""), value)

    def update_session(value, request: Request):
      sessions[request.cookies.get('user_id', "")] = value

    Context.root_block.load(resume_session, inputs=[component], outputs=[component])
    component.change(update_session, inputs=[component])

    return component