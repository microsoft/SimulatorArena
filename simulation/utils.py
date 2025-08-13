"""Tools to generate from OpenAI prompts."""

import asyncio
import logging
import os
from typing import Any, Union, List, Dict

import json
import re

import aiolimiter

import openai
from openai import AsyncAzureOpenAI, AzureOpenAI, OpenAI, AsyncOpenAI
import anthropic
from anthropic import AsyncAnthropic
from mistralai import Mistral, models
from google import genai
from google.genai import types, errors
import pydantic

from tqdm.asyncio import tqdm_asyncio

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.aio import ChatCompletionsClient as AsyncChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

import copy

from dotenv import load_dotenv
dotenv_path = os.path.expanduser('~/.env')
load_dotenv(dotenv_path)
os.environ['CURL_CA_BUNDLE'] = ''

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

why_ask_tutor_dict = {
        "L1": "I'm completely lost and don't understand the problem",
        "L2": "I understand the problem, but I don't know where to start",
        "L3": "I've written some steps, but now I'm stuck",
        "L4": "I can solve the problem, but I'm not confident in my solution",
        "Other reason": "Other reason"
}

ERROR_ERRORS_TO_MESSAGES = {
    openai.UnprocessableEntityError: "OpenAI API Invalid Request: Prompt was filtered",
    openai.RateLimitError: "OpenAI API rate limit exceeded. Sleeping for 10 seconds.",
    openai.APIStatusError: "OpenAI API Connection Error: Error Communicating with OpenAI",  # noqa E501
    openai.APITimeoutError: "OpenAI APITimeout Error: OpenAI Timeout",
    openai.InternalServerError: "OpenAI service unavailable error: {e}",
    openai.APIError: "OpenAI API error: {e}",
    openai.APIConnectionError: "OpenAI API Connection error: {e}",
    openai.BadRequestError: "OpenAI API Bad Request error: {e}",
}

import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")

import os
import json

# truncate messages
def truncate_message(message, max_tokens=200):
    if message is None:
        return ""
    message_truncated = " ".join(message.split()[-max_tokens:])
    message_truncated = "[...omitted] " + message_truncated if len(message) > len(message_truncated) else message
    return message_truncated

def construct_filename(version,
                      user_profile_version="", length_control=False, 
                      length_control_setting="", 
                      refinement=False, refinement_message_style="",
                      refinement_version="v1"):
    """
    Constructs the filename based on various parameters
    """
    
    if user_profile_version:
        if length_control:
            if "length-control" in version:
                filename = f"{version}-{length_control_setting}-up-{user_profile_version}"
            else:
                filename = f"{version}-length-control-{length_control_setting}-up-{user_profile_version}"
        else:
            filename = f"{version}-up-{user_profile_version}"
        if refinement:
            if refinement_version == "v1":
                filename += f"-ms-{refinement_message_style}-refinement"
            else:
                filename += f"-ms-{refinement_message_style}-refinement_{refinement_version}"
    else:
        if length_control:
            filename = f"{version}-{length_control_setting}"
        elif refinement:
            filename = f"{version}-ms-{refinement_message_style}-refinement"
        else:
            filename = f"{version}"

    return filename

def merge_nested_dicts(dict1, dict2):
    """
    Merges two nested dictionaries with unique ending keys.
    Args:
        dict1: First dictionary
        dict2: Second dictionary
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    def recursive_merge(current_dict, other_dict):
        for key, value in other_dict.items():
            if key not in current_dict:
                current_dict[key] = value
            elif isinstance(value, dict) and isinstance(current_dict[key], dict):
                recursive_merge(current_dict[key], value)
            else:
                # If we reach here, we're at a leaf node or there's a conflict
                # Since we guarantee unique ending keys, we keep the existing value
                print(f"Conflict at key: {key}. Keeping existing value.")
                continue
    
    recursive_merge(result, dict2)
    return result

def num_tokens_per_string(s):
    return len(enc.encode(s))

async def _throttled_openai_chat_completion_acreate(
    client: Union[AsyncAzureOpenAI, AsyncOpenAI],
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    n: int,
    json_mode: bool,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(20):
            try:
                if json_mode:
                    return await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        n=n,
                        response_format={"type": "json_object"},
                    )
                else:
                    return await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        n=n,
                    )
            except Exception as e:
                if isinstance(e, openai.UnprocessableEntityError):
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": ""
                                }
                            }
                            for _ in range(n)
                        ]
                    }
                elif isinstance(e, openai.BadRequestError):
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": ""
                                }
                            }
                            for _ in range(n)
                        ]
                    }
                # else:
                #     logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                await asyncio.sleep(10)
        return {"choices": [{"message": {"content": ""}} for _ in range(n)]}

async def generate_from_openai_chat_completion(
    full_contexts: list,
    model_name: str,
    temperature: float,
    max_tokens: int,
    top_p: float = 1.0,
    n: int = 1,
    json_mode: bool = False,
    requests_per_minute: int = 200,
    show_progress: bool = True,
) -> list[list[str]]:
    """Generate from OpenAI Chat Completion API.

    Args:
        full_contexts: List of full contexts to generate from.
        model_name: Model name.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        n: Number of responses to generate for each API call.
        top_p: Top p to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """

    client = AsyncOpenAI(
        api_key = os.environ.get("OPENAI_API_KEY"),
    )

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            client=client,
            model=model_name,
            messages=full_context,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            json_mode=json_mode,
            limiter=limiter,
        )
        for full_context in full_contexts
    ]

    if show_progress:
        responses = await tqdm_asyncio.gather(*async_responses)
    else:
        responses = await asyncio.gather(*async_responses)
    return responses

async def generate_from_azure_openai_chat_completion(
    azure_resource_name: str,
    full_contexts: list,
    model_name: str,
    temperature: float,
    max_tokens: int,
    top_p: float = 1.0,
    n: int = 1,
    json_mode: bool = False,
    requests_per_minute: int = 100,
    show_progress: bool = True,
    max_concurrent: int = 100,
) -> list[list[str]]:
    """Generate from OpenAI Chat Completion API.

    Args:
        full_contexts: List of full contexts to generate from.
        model_name: Model name.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        n: Number of responses to generate for each API call.
        top_p: Top p to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """

    client = AsyncOpenAI(
        api_key = os.environ.get("OPENAI_API_KEY"),
    )

    if model_name == "gpt-4o":
        model_name = "gpt-4o-2024-05-13"
    elif model_name == "gpt-4o-241120":
        model_name = "gpt-4o-2024-11-20"

    limiter = aiolimiter.AsyncLimiter(requests_per_minute, time_period=60)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_task(context):
        # Only allow max_concurrent tasks to run concurrently.
        async with semaphore:
            return await _throttled_openai_chat_completion_acreate(
                client=client,
                model=model_name,
                messages=context,
                temperature=temperature if temperature is not None else 0,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n,
                json_mode=json_mode,
                limiter=limiter,
            )

    # Create a task for each context.
    async_responses = [limited_task(context) for context in full_contexts]

    if show_progress:
        responses = await tqdm_asyncio.gather(*async_responses)
    else:
        responses = await asyncio.gather(*async_responses)
    return responses

MISTRAL_ERROR_MESSAGES = {
    models.HTTPValidationError: "Mistral API Validation Error: Invalid request parameters",
    models.SDKError: "Mistral API Error: {e}",
}

async def _throttled_mistral_chat_completion_acreate(
    client: Mistral,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    n: int,
    limiter: aiolimiter.AsyncLimiter,
) -> Dict[str, Any]:
    """Throttled async chat completion for Mistral API with error handling and retries."""
    async with limiter:
        for _ in range(20):  # Same retry logic as OpenAI implementation
            try:
                # Note: Mistral API doesn't support 'n' parameter directly
                # We'll need to make multiple calls if n > 1
                responses = []
                for _ in range(n):
                    response = await client.chat.complete_async(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    )
                    responses.append(response)
                
                # Format response to match OpenAI structure
                return {
                    "choices": [
                        {
                            "message": {
                                "content": resp.choices[0].message.content
                            }
                        }
                        for resp in responses
                    ]
                }

            except tuple(MISTRAL_ERROR_MESSAGES.keys()) as e:
                if isinstance(e, models.HTTPValidationError):
                    logging.warning(MISTRAL_ERROR_MESSAGES[type(e)])
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": ""
                                }
                            } for _ in range(n)
                        ]
                    }
                else:
                    # logging.warning(MISTRAL_ERROR_MESSAGES[type(e)].format(e=e))
                    await asyncio.sleep(10)
                    
        return {"choices": [{"message": {"content": ""}} for _ in range(n)]}

async def generate_from_mistral_chat_completion(
    full_contexts: List[List[Dict[str, str]]],
    model_name: str,
    temperature: float = 0.7,
    max_tokens: int = 4000,
    top_p: float = 1.0,
    n: int = 1,
    requests_per_minute: int = 20,
    requests_per_second: int = 6,  # New parameter for per-second rate limiting
    show_progress: bool = True,
    max_concurrent: int = 20,
) -> List[Dict[str, Any]]:
    """Generate from Mistral Chat Completion API.

    Args:
        full_contexts: List of message lists to generate from.
        model_name: Model name (default: "mistral-small-latest").
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        n: Number of responses to generate for each API call.
        requests_per_minute: Number of requests per minute to allow.
        requests_per_second: Number of requests per second to allow.
        show_progress: Whether to show progress bar.
        max_concurrent: Maximum number of concurrent requests.

    Returns:
        List of generated responses in OpenAI-like format.
    """
    client = Mistral(
        api_key=os.environ.get("MISTRAL_API_KEY"),
    )

    # Create both per-minute and per-second limiters
    minute_limiter = aiolimiter.AsyncLimiter(requests_per_minute, time_period=60)
    second_limiter = aiolimiter.AsyncLimiter(requests_per_second, time_period=1)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_task(context):
        # Only allow max_concurrent tasks to run concurrently.
        async with semaphore:
            # Apply both rate limits
            async with second_limiter:  # First apply per-second limit
                return await _throttled_mistral_chat_completion_acreate(
                    client=client,
                    model=model_name,
                    messages=context,
                    temperature=temperature if temperature is not None else 0,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=n,
                    limiter=minute_limiter,  # Still pass the per-minute limiter
                )

    # Create a task for each context.
    async_responses = [limited_task(context) for context in full_contexts]
    
    if show_progress:
        responses = await tqdm_asyncio.gather(*async_responses)
    else:
        responses = await asyncio.gather(*async_responses)
    return responses

ANTHROPIC_ERROR_MESSAGES = {
    anthropic.APIConnectionError: "Anthropic API Connection Error: Could not reach server",
    anthropic.RateLimitError: "Anthropic API Rate Limit Error: Backing off",
    anthropic.BadRequestError: "Anthropic API Bad Request Error: {e}",
    anthropic.AuthenticationError: "Anthropic API Authentication Error: Invalid API key",
    anthropic.PermissionDeniedError: "Anthropic API Permission Error: {e}",
    anthropic.NotFoundError: "Anthropic API Not Found Error: {e}",
    anthropic.UnprocessableEntityError: "Anthropic API Unprocessable Entity Error: {e}",
    anthropic.InternalServerError: "Anthropic API Internal Server Error: {e}",
    anthropic.APIStatusError: "Anthropic API Status Error: Non-200 status code received {e}"
}

async def _throttled_anthropic_chat_completion_acreate(
    client: AsyncAnthropic,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    n: int,
    limiter: aiolimiter.AsyncLimiter,
) -> Dict[str, Any]:
    """Throttled async chat completion for Anthropic API with error handling and retries."""

    # Make a deep copy of messages to avoid changing the original object
    messages_copy = copy.deepcopy(messages)

    system_prompt_cached = [
        {"type": "text", "text": messages_copy[0]["content"], "cache_control": {"type": "ephemeral"}},
    ]
    if messages_copy[-1]["role"] == "user":
        messages_copy[-1]["content"] = [
            {
                "type": "text",
                "text": messages_copy[-1]["content"],
                "cache_control": {"type": "ephemeral"}
            }
        ]

    async with limiter:
        for attempt in range(20):  # Retry logic
            try:
                responses = []
                # Anthropic doesn't support 'n' directly, so we make multiple calls
                for _ in range(n):
                    if messages_copy[0]["role"] == "system":
                        response = await client.messages.create(
                            model=model,
                            extra_headers={
                                "anthropic-beta": "prompt-caching-2024-07-31"
                            },
                            messages=messages_copy[1:],  # Exclude system message from messages
                            system=system_prompt_cached,
                            max_tokens=max_tokens,
                            temperature=temperature if temperature is not None else 0,
                            top_p=top_p if top_p is not None else 1.0,
                        )
                    else:
                        response = await client.messages.create(
                            model=model,
                            extra_headers={
                                "anthropic-beta": "prompt-caching-2024-07-31"
                            },
                            messages=messages_copy,
                            max_tokens=max_tokens,
                            temperature=temperature if temperature is not None else 0,
                            top_p=top_p if top_p is not None else 1.0,
                        )
                    responses.append(response)
                
                # Format response to match OpenAI-like structure for consistency
                return {
                    "choices": [
                        {
                            "message": {
                                "content": resp.content[0].text
                            }
                        }
                        for resp in responses
                    ]
                }

            except tuple(ANTHROPIC_ERROR_MESSAGES.keys()) as e:
                if isinstance(e, anthropic.RateLimitError):
                    # logging.warning(f"{ANTHROPIC_ERROR_MESSAGES[type(e)]}. Waiting {10} seconds.")
                    await asyncio.sleep(10)
                    continue
                
                elif isinstance(e, anthropic.APIConnectionError):
                    # logging.warning(f"{ANTHROPIC_ERROR_MESSAGES[type(e)]}: {e.__cause__}")
                    await asyncio.sleep(10)
                    continue
                
                elif isinstance(e, anthropic.InternalServerError):
                    # logging.warning(ANTHROPIC_ERROR_MESSAGES[type(e)].format(e=e))
                    await asyncio.sleep(10)
                    continue
                
                elif isinstance(e, (anthropic.BadRequestError, 
                                 anthropic.UnprocessableEntityError)):
                    logging.warning(ANTHROPIC_ERROR_MESSAGES[type(e)].format(e=e))
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": ""
                                }
                            } for _ in range(n)
                        ]
                    }
                
                elif isinstance(e, (anthropic.AuthenticationError, 
                                 anthropic.PermissionDeniedError)):
                    logging.error(ANTHROPIC_ERROR_MESSAGES[type(e)].format(e=e))
                    raise e
                
                else:
                    # logging.warning(ANTHROPIC_ERROR_MESSAGES[type(e)].format(e=e))
                    await asyncio.sleep(10)
                    
        return {"choices": [{"message": {"content": ""}} for _ in range(n)]}

async def generate_from_anthropic_chat_completion(
    full_contexts: List[List[Dict[str, str]]],
    model_name: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 1.0,
    n: int = 1,
    requests_per_minute: int = 100,
    show_progress: bool = True,
    max_concurrent: int = 100,
) -> List[Dict[str, Any]]:
    """Generate from Anthropic Chat Completion API.

    Args:
        full_contexts: List of message lists to generate from.
        model_name: Model name (default: "claude-3-opus-20240229").
        temperature: Temperature for generation (0-1).
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p sampling parameter.
        n: Number of responses to generate for each API call.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses in OpenAI-like format.
    """
    client = AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_task(context):
        # Only allow max_concurrent tasks to run concurrently.
        async with semaphore:
            return await _throttled_anthropic_chat_completion_acreate(
                client=client,
                model=model_name,
                messages=context,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n,
                limiter=limiter,
            )

    # Create a task for each context.
    async_responses = [limited_task(context) for context in full_contexts]
    
    if show_progress:
        responses = await tqdm_asyncio.gather(*async_responses)
    else:
        responses = await asyncio.gather(*async_responses)
    return responses

async def _throttled_azure_chat_completion_acreate(
    client: ChatCompletionsClient,
    deployment: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    n: int,
    limiter: aiolimiter.AsyncLimiter,
) -> Dict[str, Any]:
    """Throttled async chat completion for Azure AI Chat API with error handling and retries."""
    async with limiter:
        for attempt in range(20):  # Retry logic
            try:
                responses = []
                # Azure may not support 'n' directly, so we make multiple calls
                for _ in range(n):

                    # Call the Azure API
                    response = await client.complete(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        model=deployment,
                    )
                    responses.append(response)

                # Format response to match OpenAI-like structure for consistency
                return {
                    "choices": [
                        {
                            "message": {
                                "content": resp.choices[0].message.content
                            }
                        }
                        for resp in responses
                    ]
                }

            except Exception as e:
                status_code = getattr(e, 'status_code', None)
                if status_code == 429:
                    # logging.warning(f"Rate limit error: {e}. Waiting 10 seconds.")
                    await asyncio.sleep(10)
                    continue
                elif status_code == 401:
                    logging.error(f"Authentication or permission error: {e}.")
                    raise e
                elif status_code == 404:
                    logging.error(f"Resource not found: {e}.")
                    raise e
                elif "timeout" in str(e).lower() or "connection" in str(e).lower() or "cannot connect" in str(e).lower():
                    # Handle connection timeout
                    # logging.warning(f"Connection timeout: {e}. Retrying in 10 seconds...")
                    await asyncio.sleep(10)
                    continue
                else:
                    logging.error(f"Error: {e}, status code: {status_code}")
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": ""
                                }
                            }
                            for _ in range(n)
                        ]
                    }
        # After retries exhausted, return empty response
        return {"choices": [{"message": {"content": ""}} for _ in range(n)]}
    
async def generate_from_azure_chat_completion(
    model: str,
    full_contexts: List[List[Dict[str, str]]],
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 1.0,
    n: int = 1,
    requests_per_minute: int = 50,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """Generate from Azure Chat Completion API.

    Args:
        full_contexts: List of message lists to generate from.
        temperature: Temperature for generation (0-1).
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p sampling parameter.
        n: Number of responses to generate for each API call.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses in OpenAI-like format.
    """

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)

    endpoint = os.environ.get("AZURE_SLLM_ENDPOINT")
    key = os.environ.get("AZURE_SLLM_KEY")

    if model == "llama-3-1-70b":
        deployment = "Meta-Llama-3.1-70B-Instruct"
    elif model == "llama-3-1-8b":
        deployment = "Meta-Llama-3.1-8B-Instruct"
    elif model == "phi-3-medium":
        deployment = "Phi-3-medium-128k-instruct"
    elif model == "phi-3-small":
        deployment = "Phi-3-small-128k-instruct"
    elif model == "llama-3-3-70b":
        deployment = "Llama-3.3-70B-Instruct"
    elif model == "phi-4":
        deployment = "Phi-4"

    async with AsyncChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(key)) as client:
        async_responses = [
            _throttled_azure_chat_completion_acreate(
                client=client,
                deployment=deployment,
                messages=full_context,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n,
                limiter=limiter,
            )
            for full_context in full_contexts
        ]
        if show_progress:
            responses = await tqdm_asyncio.gather(*async_responses)
        else:
            responses = await asyncio.gather(*async_responses)
    return responses

# Define error messages for Gemini API
GEMINI_ERROR_MESSAGES = {
    errors.APIError: "Gemini API Error: {e}",
    errors.ClientError: "Gemini API Client Error: {e}",
    errors.ServerError: "Gemini API Server Error: {e}",
    errors.UnknownFunctionCallArgumentError: "Gemini API Unknown Function Call Argument Error: {e}",
    errors.UnsupportedFunctionError: "Gemini API Unsupported Function Error: {e}",
    errors.FunctionInvocationError: "Gemini API Function Invocation Error: {e}",
    pydantic.ValidationError: "Gemini API Validation Error: {e}"
}

async def _throttled_gemini_generate_content_acreate(
    client: genai.Client,
    model: str,
    contents: List[types.Content],
    temperature: float,
    max_tokens: int,
    top_p: float,
    n: int,
    system_prompt: str,
    limiter: aiolimiter.AsyncLimiter,
) -> Dict[str, Any]:
    """Throttled async content generation for Gemini API with error handling and retries."""
    
    # Create the generation config
    generate_content_config = types.GenerateContentConfig(
        temperature=temperature if temperature is not None else 0,
        top_p=top_p if top_p is not None else 1.0,
        max_output_tokens=max_tokens,
        response_mime_type="text/plain",
        candidate_count=n,
        system_instruction=system_prompt if system_prompt else None,
    )
    
    async with limiter:
        for attempt in range(20):  # Retry logic
            try:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                )
                
                # Format response to match OpenAI-like structure for consistency
                return {
                    "choices": [
                        {
                            "message": {
                                "content": candidate.content.parts[0].text
                            }
                        }
                        for candidate in response.candidates
                    ]
                }
                
            except errors.APIError as e:
                # Handle different error types based on error code
                if hasattr(e, 'code'):
                    if e.code == 429:  # RESOURCE_EXHAUSTED
                        # Rate limit exceeded
                        await asyncio.sleep(10)
                        continue
                        
                    elif e.code in [500, 503, 504]:  # INTERNAL, UNAVAILABLE, DEADLINE_EXCEEDED
                        # Server errors, retry after delay
                        await asyncio.sleep(10)
                        continue
                        
                    elif e.code in [400, 404]:  # INVALID_ARGUMENT, NOT_FOUND
                        # Client errors, log warning and return empty response
                        logging.warning(GEMINI_ERROR_MESSAGES[type(e)].format(e=e))
                        return {
                            "choices": [
                                {
                                    "message": {
                                        "content": ""
                                    }
                                } for _ in range(n)
                            ]
                        }
                        
                    elif e.code == 403:  # PERMISSION_DENIED
                        # Authentication errors, log error and raise exception
                        logging.error(GEMINI_ERROR_MESSAGES[type(e)].format(e=e))
                        return {
                            "choices": [
                               {
                                    "message": {
                                        "content": ""
                                    }
                                } for _ in range(n)
                            ] 
                        }
                
                # General error handling if code attribute is not available
                logging.warning(GEMINI_ERROR_MESSAGES[type(e)].format(e=e))
                await asyncio.sleep(10)
                continue
                    
            except (errors.ClientError, errors.ServerError, 
                    errors.UnknownFunctionCallArgumentError,
                    errors.UnsupportedFunctionError, 
                    errors.FunctionInvocationError,
                    pydantic.ValidationError) as e:
                # Log warning and return empty response immediately
                logging.warning(GEMINI_ERROR_MESSAGES[type(e)].format(e=e))
                return {
                    "choices": [
                        {
                            "message": {
                                "content": ""
                            }
                        } for _ in range(n)
                    ]
                }
                
        # Return empty response if all retries failed
        return {"choices": [{"message": {"content": ""}} for _ in range(n)]}

async def generate_from_gemini_api(
    full_contexts: List[List[Dict[str, str]]],
    model_name: str = "gemini-2.5-pro-exp-03-25",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 1.0,
    n: int = 1,
    requests_per_minute: int = 100,
    show_progress: bool = True,
    max_concurrent: int = 100,
) -> List[Dict[str, Any]]:
    """Generate from Google Gemini API.

    Args:
        full_contexts: List of message lists to generate from (OpenAI format: [{"role": "user", "content": "..."}]).
        model_name: Model name (default: "gemini-2.5-pro-exp-03-25").
        temperature: Temperature for generation (0-1).
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p sampling parameter.
        n: Number of responses to generate for each API call.
        requests_per_minute: Number of requests per minute to allow.
        show_progress: Whether to show progress bar.
        max_concurrent: Maximum number of concurrent requests.

    Returns:
        List of generated responses in OpenAI-like format.
    """
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_task(messages):
        # Process OpenAI-style messages to Gemini format
        contents = []
        system_prompt = None
        
        # Check for system message first (should be the first message if present)
        if messages and messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]  # Remove system message
        
        # Convert remaining messages
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            # Map OpenAI roles to Gemini roles
            if role == "user":
                gemini_role = "user"
            elif role == "assistant":
                gemini_role = "model"
            else:
                # Skip unknown roles
                continue
                
            # Create Gemini Content object
            gemini_content = types.Content(
                role=gemini_role,
                parts=[types.Part.from_text(text=content)]
            )
            contents.append(gemini_content)
        
        # Only allow max_concurrent tasks to run concurrently
        async with semaphore:
            return await _throttled_gemini_generate_content_acreate(
                client=client,
                model=model_name,
                contents=contents,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n,
                system_prompt=system_prompt,
                limiter=limiter,
            )

    # Create a task for each context
    async_responses = [limited_task(context) for context in full_contexts]
    
    if show_progress:
        responses = await tqdm_asyncio.gather(*async_responses)
    else:
        responses = await asyncio.gather(*async_responses)
    
    return responses


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
    
def extract_nested_json(text):
    # Try to find a JSON object that starts at the beginning of a line
    matches = re.finditer(r'(?m)^{.*}$', text, re.DOTALL)
    
    # Get the last match
    json_str = None
    for match in matches:
        json_str = match.group()
    
    if not json_str:
        return None
        
    # Try to parse the extracted string as JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # If parsing fails, try to clean up the string
        try:
            # Remove any leading/trailing whitespace and newlines
            json_str = json_str.strip()
            # Handle escaped newlines
            json_str = json_str.replace('\\n', '\n')
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None 

async def generate_responses_in_batch(
    full_contexts: List[List[Dict[str, str]]],
    model_name: str,
    temperature: float,
    max_tokens: int,
    n: int = 1,
    show_progress: bool = True,
) -> Union[List[str], List[List[str]]]:

    if model_name in ["claude-3-5-sonnet-20240620", "claude-3-7-sonnet-20250219"]:
        # Anthropic-like responses (dictionary-based)
        responses = await generate_from_anthropic_chat_completion(
            full_contexts, model_name, temperature=temperature, max_tokens=max_tokens, n=n, show_progress=show_progress
        )
        generated_responses = []
        for resp in responses:
            scenario_responses = []
            for i in range(n):
                # Safely get the choice for index i
                choice = resp.get('choices', [])
                if i < len(choice):
                    content = choice[i].get('message', {}).get('content', "")
                else:
                    content = ""
                try:
                    content = content.strip()
                except:
                    content = ""
                scenario_responses.append(content)
            generated_responses.append(scenario_responses)

    elif model_name in ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini", "gpt-4o-241120", "gpt-4.1-2025-04-14"]:
        # GPT-like responses (object-based)
        responses = await generate_from_azure_openai_chat_completion(
            "",
            full_contexts, model_name, temperature=temperature, max_tokens=max_tokens, n=n, show_progress=show_progress
        )
        generated_responses = []
        for resp in responses:
            scenario_responses = []
            for i in range(n):
                try:
                    content = resp.choices[i].message.content
                    content = content.strip()
                except:
                    content = ""
                scenario_responses.append(content)
            generated_responses.append(scenario_responses)

    elif model_name in ["mistral-large-latest"]:
        # Mistral-like responses (dictionary-based)
        model_name = "mistral-large-2407"
        responses = await generate_from_mistral_chat_completion(
            full_contexts, model_name, temperature=temperature, max_tokens=max_tokens, n=n, show_progress=show_progress
        )
        generated_responses = []
        for resp in responses:
            scenario_responses = []
            for i in range(n):
                choice = resp.get('choices', [])
                if i < len(choice):
                    content = choice[i].get('message', {}).get('content', "")
                else:
                    content = ""
                try:
                    content = content.strip()
                except:
                    content = ""
                scenario_responses.append(content)
            generated_responses.append(scenario_responses)

    elif model_name in ["llama-3-1-70b", "llama-3-1-8b", "phi-3-small", "phi-3-medium", "llama-3-3-70b", "phi-4"]:
        # Llama/Phi-like responses (dictionary-based)
        responses = await generate_from_azure_chat_completion(
            model_name,
            full_contexts, temperature=temperature, max_tokens=max_tokens, n=n, show_progress=show_progress
        )
        generated_responses = []
        for resp in responses:
            scenario_responses = []
            for i in range(n):
                choice = resp.get('choices', [])
                if i < len(choice):
                    content = choice[i].get('message', {}).get('content', "")
                else:
                    content = ""
                try:
                    content = content.strip()
                except:
                    content = ""
                scenario_responses.append(content)
            generated_responses.append(scenario_responses)
    elif model_name in ["gemini-2.5-pro-exp-03-25", "gemini-2.0-flash", "gemini-2.5-flash-preview-04-17"]:
        # Gemini-like responses (dictionary-based)
        responses = await generate_from_gemini_api(
            full_contexts, model_name, temperature=temperature, max_tokens=max_tokens, n=n, show_progress=show_progress
        )
        generated_responses = []
        for resp in responses:
            scenario_responses = []
            for i in range(n):
                choice = resp.get('choices', [])
                if i < len(choice):
                    content = choice[i].get('message', {}).get('content', "")
                else:
                    content = ""
                try:
                    content = content.strip()
                except:
                    content = ""
                scenario_responses.append(content)
            generated_responses.append(scenario_responses)
    else:
        # Handle other models or raise an exception
        raise ValueError(f"Unsupported model: {model_name}")

    # If n == 1, flatten the responses to a simple list of strings
    if n == 1:
        return [r[0] if r else "" for r in generated_responses]
    else:
        return generated_responses


async def simulate_conversation_in_batch_math_tutoring(
    problems: List[str],
    user_model_name: str,
    assistant_model_name: str,
    user_model_prompt_initial_query_template: str,
    user_model_prompt_template: str,
    user_temperature: float = 0.7,
    assistant_temperature: float = 0,
    max_tokens: int = 3000,
    max_turns: int = 15,
    show_progress: bool = True,
    length_control_bool: bool = False,
    length_control_list: List[str] = [],
    refinement: bool = False,
    refinement_version: str = "v1",
    user_query_style_profiles: List[str] = [],
):  
    with open(f"../prompts/refinement_{refinement_version}.txt", "r") as f:
        refinement_prompt_template = f.read()

    conversations_data = []
    for i, problem in enumerate(problems):
        # Initialize data for each conversation
        user_query_style_profile = user_query_style_profiles[i] if user_query_style_profiles else None
        length_control = length_control_list[i] if length_control_bool else None
        data = {
            'problem': problem,
            'user_query_style_profile': user_query_style_profile,
            'conversation': [],
            'conversation_history': "",
            'length_control': length_control,
            'assistant_messages': [],
            'first_query': True,
            'turns': 0,
            'finished': False,
            'over_max': False,
        }
        # Add system prompt for assistant model
        assistant_system_prompt = {
            "role": "system", 
            "content": "You are a skilled math tutor. Your goal is to help students understand and solve problems independently. Provide guidance based on their questions or mistakes. Ask questions to encourage their thinking and let students do most of the work themselves. Never give out the solution directly to students."
        }
        data['assistant_messages'].append(assistant_system_prompt)
        conversations_data.append(data)

    for turn in range(max_turns):
        # Prepare user messages for all conversations that are not finished
        user_full_contexts = []
        active_conversations = []
        for data in conversations_data:
            if data['finished'] or data['over_max']:
                continue
            if data['first_query']:
                # Use initial query template with length control
                if length_control_bool:
                    assert data['length_control'] is not None
                    user_message_content = user_model_prompt_initial_query_template.format(
                        math_problem=data['problem'], conversation_history=data['conversation_history'].strip(), length_control=data['length_control']
                    )
                else:
                    # Use initial query template
                    user_message_content = user_model_prompt_initial_query_template.format(
                        math_problem=data['problem'], conversation_history=data['conversation_history'].strip())
                data['first_query'] = False
            else:
                # Use regular prompt template
                if length_control_bool:
                    assert data['length_control'] is not None
                    user_message_content = user_model_prompt_template.format(
                        math_problem=data['problem'], conversation_history=data['conversation_history'].strip(), length_control=data['length_control']
                    )
                else:
                    user_message_content = user_model_prompt_template.format(
                        math_problem=data['problem'], conversation_history=data['conversation_history'].strip())
            user_messages = [{"role": "user", "content": user_message_content}]
            data['user_messages'] = user_messages
            user_full_contexts.append(user_messages)
            active_conversations.append(data)

        if not active_conversations:
            break  # All conversations are finished

        print(f"Generating User Queries with {user_model_name} at Turn: {turn} with length control: {length_control_bool}")

        if refinement:
            original_user_queries = await generate_responses_in_batch(
                user_full_contexts, user_model_name, user_temperature, max_tokens, show_progress=show_progress)
            
            refinement_active_conversations = []
            refinement_messages_batch = []
            for i, data in enumerate(active_conversations):
                
                if "original_user_queries" not in data:
                    data['original_user_queries'] = []

                original_user_query = original_user_queries[i]

                # Handle case where "Thought:" appears in the query
                if "Thought:" in original_user_query:
                    if "Response:" in original_user_query:
                        parts = original_user_query.split("Response:")
                    elif "Query:" in original_user_query:
                        parts = original_user_query.split("Query:")
                    else:
                        parts = original_user_query.split("Message:")
                    if len(parts) > 1:
                        original_user_query = parts[1].strip()

                data['original_user_queries'].append(original_user_query)

                if "terminate conversation" in original_user_query.lower() or not original_user_query:
                    data['conversation'].append(("user", original_user_query))
                    data['finished'] = True
                    continue

                refinement_active_conversations.append(data)

                conversation_history = data['conversation_history'].strip() if data['conversation_history'] else "<empty>"
                refinement_prompt = refinement_prompt_template.format(
                    user_profile=data['user_query_style_profile'],
                    math_problem=data['problem'],
                    conversation_history=conversation_history,
                    original_user_message=original_user_query,
                    length_control=data['length_control'],
                )

                refinement_message = [{"role": "user", "content": refinement_prompt}]
                refinement_messages_batch.append(refinement_message)

            active_conversations = refinement_active_conversations

            print(f"Generating <Refined> User Queries with {user_model_name} at Turn: {turn}")

            user_queries = await generate_responses_in_batch(
                refinement_messages_batch,
                user_model_name,
                user_temperature,
                max_tokens,
                show_progress=show_progress
            )
        else:
            # Generate user queries in batch using the combined helper function
            user_queries = await generate_responses_in_batch(
                user_full_contexts, user_model_name, user_temperature, max_tokens, show_progress=show_progress)

        # Update conversations with user queries
        for data, user_query in zip(active_conversations, user_queries):
            data['conversation'].append(("user", user_query))

            # Process user query, extract query if "Thought:" in user_query
            if "Thought:" in user_query:
                try:
                    if "Response:" in user_query:
                        query = user_query.split("Response:")[1].strip()
                    elif "Query:" in user_query:
                        query = user_query.split("Query:")[1].strip()
                    else:
                        query = user_query.split("Message:")[1].strip()
                except:
                    print(f"No query found in user query: {user_query}")
                    data['finished'] = True
                    continue
            else:
                query = user_query

            if "terminate conversation" in user_query.lower() or not user_query:
                data['finished'] = True
                continue

            # Prepare assistant messages
            if len(data['assistant_messages']) == 1:
                # First turn
                first_turn_user_query = f"Here is the problem that you will tutor me on:\n{data['problem'].strip()}\n\n{query}"
                data['assistant_messages'].append({"role": "user", "content": first_turn_user_query})
                data['first_query_content'] = query
            else:
                data['assistant_messages'].append({"role": "user", "content": query})


        active_conversations = [data for data in active_conversations if not data['finished']]

        if not active_conversations:
            break

        # Prepare assistant messages for all active conversations
        assistant_full_contexts = [data['assistant_messages'] for data in active_conversations]

        print(f"Generating Assistant Responses with {assistant_model_name} at Turn: {turn}")

        # Generate assistant responses in batch using the combined helper function
        assistant_responses = await generate_responses_in_batch(
            assistant_full_contexts, assistant_model_name, assistant_temperature, max_tokens, show_progress=show_progress)

        # Update conversations with assistant responses
        for data, assistant_response in zip(active_conversations, assistant_responses):
            data['conversation'].append(("assistant", assistant_response))
            last_user_message = data['assistant_messages'][-1]['content']
            if len(data['assistant_messages']) == 2:
                last_user_message = data['first_query_content']
            data['conversation_history'] += f"- You: {last_user_message}\n- AI Tutor: {assistant_response}\n"
            data['assistant_messages'].append({"role": "assistant", "content": assistant_response})

            data['turns'] += 1

            if not assistant_response:
                data['finished'] = True

            # Check if max turns reached
            if data['turns'] >= max_turns:
                data['over_max'] = True

    return conversations_data


async def simulate_conversation_with_user_profile_in_batch_math_tutoring(
    problems: List[str],
    user_profiles: List[str],
    user_model_name: str,
    assistant_model_name: str,
    user_model_prompt_initial_query_template: str,
    user_model_prompt_template: str,
    user_temperature: float = 0.7,
    assistant_temperature: float = 0,
    max_tokens: int = 3000,
    max_turns: int = 15,
    length_control_bool: bool = False,
    length_control_list: List[str] = [],
    refinement: bool = False,
    refinement_version: str = "v1",
    user_query_style_profiles: List[str] = [],
    n: int = 1,
    show_progress: bool = True,
    focus_feature_texts: List[str] = [],
):  

    # load refinement prompt template
    with open(f"../prompts/refinement_{refinement_version}.txt", "r") as f:
        refinement_prompt_template = f.read()

    conversations_data = []
    for i, problem in enumerate(problems):
        user_profile = user_profiles[i]
        length_control = length_control_list[i] if length_control_bool else None
        user_query_style_profile = user_query_style_profiles[i] if user_query_style_profiles else None
        focus_feature_text = focus_feature_texts[i] if focus_feature_texts else None
        # Initialize data for each conversation
        data = {
            'problem': problem,
            'user_profile': user_profile,
            'user_query_style_profile': user_query_style_profile,
            'focus_feature_text': focus_feature_text,
            'length_control': length_control,
            'conversation': [],
            'conversation_history': "",
            'assistant_messages': [],
            'first_query': True,
            'turns': 0,
            'finished': False,
            'over_max': False,
        }
        # Add system prompt for assistant model
        assistant_system_prompt = {
            "role": "system", 
            "content": "You are a skilled math tutor. Your goal is to help students understand and solve problems independently. Provide guidance based on their questions or mistakes. Ask questions to encourage their thinking and let students do most of the work themselves. Never give out the solution directly to students."
        }
        data['assistant_messages'].append(assistant_system_prompt)
        conversations_data.append(data)

    for turn in range(max_turns):
        # Prepare user messages for all conversations that are not finished
        user_full_contexts = []
        active_conversations = []
        for data in conversations_data:
            if data['finished'] or data['over_max']:
                continue
            if data['first_query']:
                if length_control_bool:
                    assert data['length_control'] is not None
                    # Use initial query template with length control
                    user_message_content = user_model_prompt_initial_query_template.format(
                        user_profile=data["user_profile"], math_problem=data['problem'], conversation_history=data['conversation_history'].strip(), length_control=data['length_control'], focus_feature_text=data['focus_feature_text']
                    )
                else:
                    # Use initial query template
                    user_message_content = user_model_prompt_initial_query_template.format(
                        user_profile=data["user_profile"], math_problem=data['problem'], focus_feature_text=data['focus_feature_text'])
                data['first_query'] = False
            else:
                if length_control_bool:
                    assert data['length_control'] is not None
                    # Use regular prompt template with length control
                    user_message_content = user_model_prompt_template.format(
                        user_profile=data["user_profile"],
                        math_problem=data['problem'], conversation_history=data['conversation_history'].strip(), length_control=data['length_control'], focus_feature_text=data['focus_feature_text'])
                else:
                    # Use regular prompt template
                    user_message_content = user_model_prompt_template.format(
                        user_profile=data["user_profile"],
                        math_problem=data['problem'], conversation_history=data['conversation_history'].strip(), focus_feature_text=data['focus_feature_text'])
            
            user_messages = [{"role": "user", "content": user_message_content}]

            data['user_messages'] = user_messages
            user_full_contexts.append(user_messages)
            active_conversations.append(data)

        if not active_conversations:
            break  # All conversations are finished

        print(f"Generating User Queries with {user_model_name} with User profile with length control as {length_control_bool} at Turn: {turn}")
        
        if refinement:
            original_user_queries = await generate_responses_in_batch(
                user_full_contexts, user_model_name, user_temperature, max_tokens, show_progress=show_progress)
            
            refinement_active_conversations = []
            refinement_messages_batch = []
            for i, data in enumerate(active_conversations):
                
                if "original_user_queries" not in data:
                    data['original_user_queries'] = []

                original_user_query = original_user_queries[i]

                # Handle case where "Thought:" appears in the query
                if "Thought:" in original_user_query:
                    if "Response:" in original_user_query:
                        parts = original_user_query.split("Response:")
                    elif "Query:" in original_user_query:
                        parts = original_user_query.split("Query:")
                    else:
                        parts = original_user_query.split("Message:")
                    if len(parts) > 1:
                        original_user_query = parts[1].strip()

                data['original_user_queries'].append(original_user_query)

                if "terminate conversation" in original_user_query.lower() or not original_user_query:
                    data['conversation'].append(("user", original_user_query))
                    data['finished'] = True
                    continue

                refinement_active_conversations.append(data)

                conversation_history = data['conversation_history'].strip() if data['conversation_history'] else "<empty>"
                refinement_prompt = refinement_prompt_template.format(
                    user_profile=data['user_query_style_profile'],
                    math_problem=data['problem'],
                    conversation_history=conversation_history,
                    original_user_message=original_user_query,
                    length_control=data['length_control'],
                )
                
                refinement_message = [{"role": "user", "content": refinement_prompt}]
                refinement_messages_batch.append(refinement_message)

            active_conversations = refinement_active_conversations

            print(f"Generating <Refined> version {refinement_version} User Queries with {user_model_name} at Turn: {turn}")

            user_queries = await generate_responses_in_batch(
                refinement_messages_batch,
                user_model_name,
                user_temperature,
                max_tokens,
                show_progress=show_progress
            )
        else:
            # Generate user queries in batch using the combined helper function
            user_queries = await generate_responses_in_batch(
                user_full_contexts, user_model_name, user_temperature, max_tokens, show_progress=show_progress)

        # Update conversations with user queries
        for data, user_query in zip(active_conversations, user_queries):
            data['conversation'].append(("user", user_query))

            # Process user query, extract query if "Thought:" in user_query
            if "Thought:" in user_query:
                try:
                    if "Response:" in user_query:
                        query = user_query.split("Response:")[1].strip()
                    elif "Query:" in user_query:
                        query = user_query.split("Query:")[1].strip()
                    else:
                        query = user_query.split("Message:")[1].strip()
                except:
                    print(f"No query found in user query: {user_query}")
                    data['finished_by_error'] = True
                    continue
            else:
                query = user_query

            if "terminate conversation" in user_query.lower() or not user_query:
                data['finished'] = True
                continue

            # Prepare assistant messages
            if len(data['assistant_messages']) == 1:
                # First turn
                first_turn_user_query = f"Here is the problem that you will tutor me on:\n{data['problem'].strip()}\n\n{query}"
                data['assistant_messages'].append({"role": "user", "content": first_turn_user_query})
                data['first_query_content'] = query
            else:
                data['assistant_messages'].append({"role": "user", "content": query})


        active_conversations = [data for data in active_conversations if not data['finished']]

        if not active_conversations:
            break

        # Prepare assistant messages for all active conversations
        assistant_full_contexts = [data['assistant_messages'] for data in active_conversations]

        print(f"Generating Assistant Responses with {assistant_model_name} at Turn: {turn}")

        # Generate assistant responses in batch using the combined helper function
        assistant_responses = await generate_responses_in_batch(
            assistant_full_contexts, assistant_model_name, assistant_temperature, max_tokens, show_progress=show_progress)

        # Update conversations with assistant responses
        for data, assistant_response in zip(active_conversations, assistant_responses):
            data['conversation'].append(("assistant", assistant_response))
            last_user_message = data['assistant_messages'][-1]['content']
            if len(data['assistant_messages']) == 2:
                last_user_message = data['first_query_content']
            data['conversation_history'] += f"- You: {last_user_message}\n- AI Tutor: {assistant_response}\n"
            data['assistant_messages'].append({"role": "assistant", "content": assistant_response})

            data['turns'] += 1

            if not assistant_response:
                data['finished'] = True

            # Check if max turns reached
            if data['turns'] >= max_turns:
                data['over_max'] = True

    return conversations_data

    # Define the fixed assistant (tutor) system prompt
    assistant_system_prompt = {
        "role": "system", 
        "content": "You are a skilled math tutor. Your goal is to help students understand and solve problems independently. Provide guidance based on their questions or mistakes. Ask questions to encourage their thinking and let students do most of the work themselves. Never give out the solution directly to students."
    }

    conversations_data = []
    for i, problem in enumerate(problems):
        # Initialize data for each conversation
        length_control = length_control_list[i] if length_control_bool else None
        data = {
            'problem': problem,
            'conversation': [],
            'length_control': length_control,
            'user_messages': [],  # Will store full message history for user model
            'assistant_messages': [],  # Will store full message history for assistant model
            'turns': 0,
            'finished': False,
            'over_max': False,
        }

        if length_control:
            user_system_prompt = user_system_prompt_template.format(
                math_problem=problem, length_control=length_control)
        else:
            user_system_prompt = user_system_prompt_template.format(math_problem=problem)

        data["user_system_prompt"] = user_system_prompt
        
        # Initialize system prompts for both models
        # For user (student) model, include the math problem in system prompt
        data['user_messages'].append({
            "role": "system",
            "content": f"{user_system_prompt}"
        })

        # Add the initial assistant message to user's context
        data['user_messages'].append({
            "role": "user",
            "content": "I am your math tutor. What can I help you?"
        })
        
        # For assistant (tutor) model
        data['assistant_messages'].append(assistant_system_prompt)
        
        conversations_data.append(data)

    for turn in range(max_turns):
        # Prepare contexts for active conversations
        active_conversations = []
        assistant_full_contexts = []
        
        for data in conversations_data:
            if data['finished'] or data['over_max']:
                continue
                
            active_conversations.append(data)
            assistant_full_contexts.append(data['assistant_messages'])

        if not active_conversations:
            break  # All conversations are finished

        print(f"Generating User Queries with {user_model_name} at Turn: {turn}")

        # if turn==3:
        #     print(active_conversations[0]["user_messages"])
        #     print("###########################")
        #     print(active_conversations[0]["assistant_messages"])
        #     print("---------------------------")
        #     exit()

        # Generate user responses based on full conversation context
        user_queries = await generate_responses_in_batch(
            [data['user_messages'] for data in active_conversations],
            user_model_name,
            user_temperature,
            max_tokens,
            show_progress=show_progress
        )

        # Update conversations with user queries
        for data, user_query in zip(active_conversations, user_queries):
            data['conversation'].append(("user", user_query))
            if "Thought:" in user_query:
                try:
                    if "Response:" in user_query:
                        query = user_query.split("Response:")[1].strip()
                    elif "Query:" in user_query:
                        query = user_query.split("Query:")[1].strip()
                    else:
                        query = user_query.split("Message:")[1].strip()
                except:
                    print(f"No query found in user query: {user_query}")
                    data['finished'] = True
                    continue
            else:
                query = user_query

            if "terminate conversation" in user_query.lower() or not user_query:
                data['finished'] = True
                continue

            # Prepare assistant messages
            if len(data['assistant_messages']) == 1:
                # First turn
                first_turn_user_query = f"Here is the problem that you will tutor me on:\n{data['problem'].strip()}\n\n{query}"
                data['assistant_messages'].append({"role": "user", "content": first_turn_user_query})
                data['first_query_content'] = query
            else:
                data['assistant_messages'].append({"role": "user", "content": query})

            # Add user message
            user_message = {"role": "assistant", "content": user_query if user_thought else query}
            data['user_messages'].append(user_message)

        # Filter out finished conversations
        active_conversations = [data for data in active_conversations if not data['finished']]
        if not active_conversations:
            break

        print(f"Generating Assistant Responses with {assistant_model_name} at Turn: {turn}")

        # Generate assistant responses using full conversation context
        assistant_responses = await generate_responses_in_batch(
            [data['assistant_messages'] for data in active_conversations],
            assistant_model_name,
            assistant_temperature,
            max_tokens,
            show_progress=show_progress
        )

        # Update conversations with assistant responses
        for data, assistant_response in zip(active_conversations, assistant_responses):

            # Add assistant response to both conversation histories
            data['user_messages'].append({"role": "user", "content": assistant_response})
            data['assistant_messages'].append({"role": "assistant", "content": assistant_response})
            data['conversation'].append(("assistant", assistant_response))
            
            data['turns'] += 1
            if not assistant_response:
                data['finished'] = True
                continue

            # Check if max turns reached
            if data['turns'] >= max_turns:
                data['over_max'] = True

    return conversations_data


async def simulate_conversation_in_batch_document_creation(
    document_types: List[str],
    intents: List[str],
    backgrounds: List[str],
    user_model_name: str,
    assistant_model_name: str,
    user_model_prompt_initial_query_template: str,
    user_model_prompt_template: str,
    user_temperature: float = 0.7,
    assistant_temperature: float = 0,
    max_tokens: int = 2000,
    max_turns: int = 15,
    show_progress: bool = True,
    length_control_bool: bool = False,
    length_control_list: List[str] = [],
    refinement: bool = False,
    refinement_version: str = "v1",
    user_query_style_profiles: List[str] = [],
):  
    with open(f"../prompts/refinement_{refinement_version}.txt", "r") as f:
        refinement_prompt_template = f.read()

    conversations_data = []
    for i, document_type in enumerate(document_types):
        # Initialize data for each conversation
        user_query_style_profile = user_query_style_profiles[i] if user_query_style_profiles else None
        length_control = length_control_list[i] if length_control_bool else None
        data = {
            'document_type': document_type,
            'intent': intents[i],
            'background': backgrounds[i],
            'user_query_style_profile': user_query_style_profile,
            'conversation': [],
            'conversation_history': "",
            'length_control': length_control,
            'assistant_messages': [],
            'first_query': True,
            'turns': 0,
            'finished': False,
            'over_max': False,
        }
        # Add system prompt for assistant model
        assistant_system_prompt = {
            "role": "system", 
            "content": "You are a skilled writing assistant. Your role is to help users create and edit documents that should be under 600 words by following their specific instructions and requirements."
        }
        data['assistant_messages'].append(assistant_system_prompt)
        conversations_data.append(data)

    for turn in range(max_turns):
        # Prepare user messages for all conversations that are not finished
        user_full_contexts = []
        active_conversations = []
        for data in conversations_data:
            if data['finished'] or data['over_max']:
                continue
            if data['first_query']:
                # Use initial query template with length control
                if length_control_bool:
                    assert data['length_control'] is not None
                    user_message_content = user_model_prompt_initial_query_template.format(
                        document_type=data['document_type'], intent=data['intent'], pre_writing_materials=data["background"], conversation_history=data['conversation_history'].strip(), length_control=data['length_control']
                    )
                else:
                    # Use initial query template
                    user_message_content = user_model_prompt_initial_query_template.format(
                        document_type=data['document_type'], intent=data['intent'], pre_writing_materials=data["background"], conversation_history=data['conversation_history'].strip())
                data['first_query'] = False
            else:
                # Use regular prompt template
                if length_control_bool:
                    assert data['length_control'] is not None
                    user_message_content = user_model_prompt_template.format(
                        document_type=data['document_type'], intent=data['intent'], pre_writing_materials=data["background"], conversation_history=data['conversation_history'].strip(), length_control=data['length_control']
                    )
                else:
                    user_message_content = user_model_prompt_template.format(
                        document_type=data['document_type'], intent=data['intent'], pre_writing_materials=data["background"], conversation_history=data['conversation_history'].strip())


            user_messages = [{"role": "user", "content": user_message_content}]
            data['user_messages'] = user_messages
            user_full_contexts.append(user_messages)
            active_conversations.append(data)

        if not active_conversations:
            break  # All conversations are finished

        print(f"Generating User Queries with {user_model_name} at Turn: {turn} with length control: {length_control_bool}")

        if refinement:
            original_user_queries = await generate_responses_in_batch(
                user_full_contexts, user_model_name, user_temperature, max_tokens, show_progress=show_progress)
            
            refinement_active_conversations = []
            refinement_messages_batch = []
            for i, data in enumerate(active_conversations):
                
                if "original_user_queries" not in data:
                    data['original_user_queries'] = []

                original_user_query = original_user_queries[i]

                # Handle case where "Thought:" appears in the query
                if "Thought:" in original_user_query:
                    parts = original_user_query.split("Message:")
                    if len(parts) > 1:
                        original_user_query = parts[1].strip()

                data['original_user_queries'].append(original_user_query)

                if "terminate conversation" in original_user_query.lower() or not original_user_query:
                    data['conversation'].append(("user", original_user_query))
                    data['finished'] = True
                    continue

                refinement_active_conversations.append(data)

                conversation_history = data['conversation_history'].strip() if data['conversation_history'] else "<empty>"
                refinement_prompt = refinement_prompt_template.format(
                    user_message_style=data['user_query_style_profile'],
                    document_type=data['document_type'],
                    intent=data['intent'],
                    conversation_history=conversation_history,
                    original_user_message=original_user_query,
                    length_control=data['length_control'],
                )

                refinement_message = [{"role": "user", "content": refinement_prompt}]
                refinement_messages_batch.append(refinement_message)

            active_conversations = refinement_active_conversations

            print(f"Generating <Refined> version {refinement_version} User Queries with {user_model_name} at Turn: {turn}")

            user_queries = await generate_responses_in_batch(
                refinement_messages_batch,
                user_model_name,
                user_temperature,
                max_tokens,
                show_progress=show_progress
            )
        else:
            # Generate user queries in batch using the combined helper function
            user_queries = await generate_responses_in_batch(
                user_full_contexts, user_model_name, user_temperature, max_tokens, show_progress=show_progress)

        # Update conversations with user queries
        for data, user_query in zip(active_conversations, user_queries):
            data['conversation'].append(("user", user_query))

            # Process user query, extract query if "Thought:" in user_query
            if "Thought:" in user_query:
                try:
                    query = user_query.split("Message:")[1].strip()
                except:
                    print(f"No query found in user query: {user_query}")
                    data['finished'] = True
                    continue
            else:
                query = user_query

            if not user_query:
                print(f"User query is empty")
                data['finished'] = True
                continue

            if "terminate conversation" in user_query.lower():
                print(f"User query contains 'terminate conversation'")
                data['finished'] = True
                continue

            data['assistant_messages'].append({"role": "user", "content": query})

        active_conversations = [data for data in active_conversations if not data['finished']]

        if not active_conversations:
            break

        # Prepare assistant messages for all active conversations
        assistant_full_contexts = [data['assistant_messages'] for data in active_conversations]

        print(f"Generating Assistant Responses with {assistant_model_name} at Turn: {turn}")

        # Generate assistant responses in batch using the combined helper function
        assistant_responses = await generate_responses_in_batch(
            assistant_full_contexts, assistant_model_name, assistant_temperature, max_tokens, show_progress=show_progress)

        # Update conversations with assistant responses
        for data, assistant_response in zip(active_conversations, assistant_responses):
            data['conversation'].append(("assistant", assistant_response))
            last_user_message = data['assistant_messages'][-1]['content']
            data['conversation_history'] += f"- You: {last_user_message}\n- AI Writing Assistant: {assistant_response}\n"
            data['assistant_messages'].append({"role": "assistant", "content": assistant_response})

            data['turns'] += 1

            if not assistant_response:
                data['finished'] = True

            # Check if max turns reached
            if data['turns'] >= max_turns:
                data['over_max'] = True

    return conversations_data

async def simulate_conversation_with_user_profile_in_batch_document_creation(
    document_types: List[str],
    intents: List[str],
    backgrounds: List[str],
    user_profiles: List[str],
    user_model_name: str,
    assistant_model_name: str,
    user_model_prompt_initial_query_template: str,
    user_model_prompt_template: str,
    user_temperature: float = 0.7,
    assistant_temperature: float = 0,
    max_tokens: int = 2000,
    max_turns: int = 15,
    length_control_bool: bool = False,
    length_control_list: List[str] = [],
    refinement: bool = False,
    refinement_version: str = "v1",
    user_query_style_profiles: List[str] = [],
    n: int = 1,
    show_progress: bool = True,
    focus_feature_texts: List[str] = [],
):  

    # load refinement prompt template
    with open(f"../prompts/refinement_{refinement_version}.txt", "r") as f:
        refinement_prompt_template = f.read()


    conversations_data = []
    for i, document_type in enumerate(document_types):
        user_profile = user_profiles[i]
        length_control = length_control_list[i] if length_control_bool else None
        user_query_style_profile = user_query_style_profiles[i] if user_query_style_profiles else None
        focus_feature_text = focus_feature_texts[i] if focus_feature_texts else None
        # Initialize data for each conversation
        data = {
            'document_type': document_type,
            'intent': intents[i],
            'background': backgrounds[i],
            'user_profile': user_profile,
            'user_query_style_profile': user_query_style_profile,
            'focus_feature_text': focus_feature_text,
            'length_control': length_control,
            'conversation': [],
            'conversation_history': "",
            'assistant_messages': [],
            'first_query': True,
            'turns': 0,
            'finished': False,
            'over_max': False,
        }
        # Add system prompt for assistant model
        assistant_system_prompt = {
            "role": "system", 
            "content": "You are a skilled writing assistant. Your role is to help users create and edit documents that should be under 600 words by following their specific instructions and requirements."
        }
        data['assistant_messages'].append(assistant_system_prompt)
        conversations_data.append(data)

    for turn in range(max_turns):
        # Prepare user messages for all conversations that are not finished
        user_full_contexts = []
        active_conversations = []
        for data in conversations_data:
            if data['finished'] or data['over_max']:
                continue
            if data['first_query']:
                if length_control_bool:
                    assert data['length_control'] is not None
                    # Use initial query template with length control
                    user_message_content = user_model_prompt_initial_query_template.format(
                        user_profile=data["user_profile"], document_type=data['document_type'], 
                        intent=data['intent'], pre_writing_materials=data["background"], 
                        conversation_history=data['conversation_history'].strip(), length_control=data['length_control'],
                        focus_feature_text=data['focus_feature_text']
                    )
                else:
                    # Use initial query template
                    user_message_content = user_model_prompt_initial_query_template.format(
                        user_profile=data["user_profile"], document_type=data['document_type'], 
                        intent=data['intent'], pre_writing_materials=data["background"], 
                        conversation_history=data['conversation_history'].strip(),
                        focus_feature_text=data['focus_feature_text']
                    )
                data['first_query'] = False
            else:
                if length_control_bool:
                    assert data['length_control'] is not None
                    # Use regular prompt template with length control
                    user_message_content = user_model_prompt_template.format(
                        user_profile=data["user_profile"], document_type=data['document_type'], 
                        intent=data['intent'], pre_writing_materials=data["background"], 
                        conversation_history=data['conversation_history'].strip(), length_control=data['length_control'],
                        focus_feature_text=data['focus_feature_text']
                    )
                else:
                    # Use regular prompt template
                    user_message_content = user_model_prompt_template.format(
                        user_profile=data["user_profile"], document_type=data['document_type'], 
                        intent=data['intent'], pre_writing_materials=data["background"], 
                        conversation_history=data['conversation_history'].strip(),
                        focus_feature_text=data['focus_feature_text']
                    )
                    
            user_messages = [{"role": "user", "content": user_message_content}]

            data['user_messages'] = user_messages
            user_full_contexts.append(user_messages)
            active_conversations.append(data)

        if not active_conversations:
            break  # All conversations are finished

        print(f"Generating User Queries with {user_model_name} with User profile with length control as {length_control_bool} at Turn: {turn}")

        if refinement:
            original_user_queries = await generate_responses_in_batch(
                user_full_contexts, user_model_name, user_temperature, max_tokens, show_progress=show_progress)
            
            refinement_active_conversations = []
            refinement_messages_batch = []
            for i, data in enumerate(active_conversations):
                
                if "original_user_queries" not in data:
                    data['original_user_queries'] = []

                original_user_query = original_user_queries[i]

                # Handle case where "Thought:" appears in the query
                if "Thought:" in original_user_query:
                    parts = original_user_query.split("Message:")
                    if len(parts) > 1:
                        original_user_query = parts[1].strip()

                data['original_user_queries'].append(original_user_query)

                if "terminate conversation" in original_user_query.lower() or not original_user_query:
                    data['conversation'].append(("user", original_user_query))
                    data['finished'] = True
                    continue

                refinement_active_conversations.append(data)

                conversation_history = data['conversation_history'].strip() if data['conversation_history'] else "<empty>"
                refinement_prompt = refinement_prompt_template.format(
                    user_message_style=data['user_query_style_profile'],
                    document_type=data['document_type'],
                    intent=data['intent'],
                    conversation_history=conversation_history,
                    original_user_message=original_user_query,
                    length_control=data["length_control"]
                )

                refinement_message = [{"role": "user", "content": refinement_prompt}]
                refinement_messages_batch.append(refinement_message)

            active_conversations = refinement_active_conversations

            print(f"Generating <Refined> version {refinement_version} User Queries with {user_model_name} at Turn: {turn}")

            user_queries = await generate_responses_in_batch(
                refinement_messages_batch,
                user_model_name,
                user_temperature,
                max_tokens,
                show_progress=show_progress
            )
        else:
            # Generate user queries in batch using the combined helper function
            user_queries = await generate_responses_in_batch(
                user_full_contexts, user_model_name, user_temperature, max_tokens, show_progress=show_progress)

        # Update conversations with user queries
        for data, user_query in zip(active_conversations, user_queries):
            data['conversation'].append(("user", user_query))

            # Process user query, extract query if "Thought:" in user_query
            if "Thought:" in user_query:
                try:
                    query = user_query.split("Message:")[1].strip()
                except:
                    print(f"No query found in user query: {user_query}")
                    data['finished_by_error'] = True
                    continue
            else:
                query = user_query

            if "terminate conversation" in user_query.lower() or not user_query:
                data['finished'] = True
                continue

            data['assistant_messages'].append({"role": "user", "content": query})


        active_conversations = [data for data in active_conversations if not data['finished']]

        if not active_conversations:
            break

        # Prepare assistant messages for all active conversations
        assistant_full_contexts = [data['assistant_messages'] for data in active_conversations]

        print(f"Generating Assistant Responses with {assistant_model_name} at Turn: {turn}")

        # Generate assistant responses in batch using the combined helper function
        assistant_responses = await generate_responses_in_batch(
            assistant_full_contexts, assistant_model_name, assistant_temperature, max_tokens, show_progress=show_progress)

        # Update conversations with assistant responses
        for data, assistant_response in zip(active_conversations, assistant_responses):
            data['conversation'].append(("assistant", assistant_response))
            last_user_message = data['assistant_messages'][-1]['content']
            data['conversation_history'] += f"- You: {last_user_message}\n- AI Writing Assistant: {assistant_response}\n"
            data['assistant_messages'].append({"role": "assistant", "content": assistant_response})

            data['turns'] += 1

            if not assistant_response:
                data['finished'] = True

            # Check if max turns reached
            if data['turns'] >= max_turns:
                data['over_max'] = True

    return conversations_data