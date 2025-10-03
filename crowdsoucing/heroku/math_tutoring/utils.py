import urllib.request
import json
import os
import ssl
from promptflow.core import tool
import time

from gradio.context import Context
from gradio import Request
import gradio as gr


from dotenv import load_dotenv
dotenv_path = os.path.expanduser('~/.env')
load_dotenv(dotenv_path)


model_name_deployment_map = {
    "llama-3-70b-instruct": {
        "url": "https://gcr-llama3-70b-instruct.westus3.inference.ml.azure.com/score",
        "deployment": "meta-llama-3-70b-instruct-4",
        "api_key": os.environ["GCR_LLAMA3_70B_INSTRUCT_KEY"]
    },
    "llama-3-8b-instruct": {
        "url": "https://gcr-llama-3-8b-instruct.westus3.inference.ml.azure.com/score",
        "deployment": "meta-llama-3-8b-instruct-4",
        "api_key": os.environ["GCR_LLAMA3_8B_INSTRUCT_KEY"]
    },
    "phi-3-small-128k-instruct": {
        "url": "https://gcr-phi-3-small-128k-instruct.westus3.inference.ml.azure.com/score",
        "deployment": "phi-3-small-128k-instruct-1",
        "api_key": os.environ["GCR_PHI3_SMALL_128K_INSTRUCT_KEY"]
    },
    "phi-3-medium-128k-instruct": {
        "url": "https://gcr-phi-3-medium-128k-instruct.westus3.inference.ml.azure.com/score",
        "deployment": "phi-3-medium-128k-instruct-1",
        "api_key": os.environ["GCR_PHI3_MEDIUM_128K_INSTRUCT_KEY"]
    },
}

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context


def create(model_name, history):
  allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

  # Request data goes here
  # The example below assumes JSON formatting which may be updated
  # depending on the format your endpoint expects.
  # More information can be found here:
  # https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
  data = {
      
    "input_data": {
      "input_string": history,
      "parameters": {
        "temperature": 0,
        "max_new_tokens": 2000
      }
    }
    
  }

  body = str.encode(json.dumps(data))

  url = model_name_deployment_map[model_name]["url"]
  api_key = model_name_deployment_map[model_name]["api_key"]
  # Replace this with the primary/secondary key, AMLToken, or Microsoft Entra ID token for the endpoint
  if not api_key:
      raise Exception("A key should be provided to invoke the endpoint")

  # The azureml-model-deployment header will force the request to go to a specific deployment.
  # Remove this header to have the request observe the endpoint traffic rules
  headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': model_name_deployment_map[model_name]["deployment"] }

  req = urllib.request.Request(url, body, headers)

  @tool
  def generator(paragraph: str):
      for word in paragraph.split():
        yield word + " "
        time.sleep(0.015)

  try:
      response = urllib.request.urlopen(req)
      result = response.read()
      output = eval(result.decode("utf-8"))["output"]
      return generator(output)
  except urllib.error.HTTPError as error:
      print("The request failed with status code: " + str(error.code))

      # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
      print(error.info())
      return error.info()
  

def rreplace(s, old, new, occurrence):
  li = s.rsplit(old, occurrence)
  return new.join(li)
      

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