#!/usr/bin/env python3
"""Reusable helper for calling OpenAI / Azure OpenAI chat completions."""

import os
import sys
import time
from typing import Any, Iterable, Mapping, MutableMapping

import openai
from openai import AzureOpenAI, OpenAI

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_P = 0.95
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_PRESENCE_PENALTY = 0.0
MAX_RETRIES = 10
RETRY_DELAY_SECONDS = 10


class MissingConfiguration(ValueError):
    """Raised when required environment variables are missing."""


def _prepare_messages(
    messages: Iterable[Mapping[str, Any]], system_prompt: str
) -> list[MutableMapping[str, Any]]:
    message_list = [dict(msg) for msg in messages]
    if not message_list:
        raise ValueError("messages must contain at least one message")
    if message_list[0].get("role") != "system":
        message_list.insert(0, {"role": "system", "content": system_prompt})
    return message_list


def _has_image_content(message_list: Iterable[Mapping[str, Any]]) -> bool:
    for message in message_list:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "image_url":
                    return True
    return False


def _build_client(model_type: str):
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    if azure_endpoint and azure_key:
        deployment_name = model_type or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        if not deployment_name:
            raise MissingConfiguration(
                "Provide a deployment name when using Azure OpenAI "
                "(pass model_type or set AZURE_OPENAI_DEPLOYMENT)."
            )
        client = AzureOpenAI(azure_endpoint=azure_endpoint, api_key=azure_key, api_version=azure_api_version)
        target_model = deployment_name
    else:
        if not model_type:
            raise MissingConfiguration("model_type is required when AZURE_OPENAI_ENDPOINT is not set.")
        client = OpenAI()
        target_model = model_type
    return client, target_model


def _allows_sampling_params(model_name: str) -> bool:
    lowered = model_name.lower()
    reasoning_prefixes = ("gpt-5", "o3", "o4")
    return not any(lowered.startswith(prefix) for prefix in reasoning_prefixes)


def call_chat_completion(
    model_type: str,
    messages: Iterable[Mapping[str, Any]],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float | None = None,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Send a chat completion request and return the assistant response text."""
    client, target_model = _build_client(model_type)
    message_list = _prepare_messages(messages, system_prompt)
    has_image = _has_image_content(message_list)

    payload: dict[str, Any] = {
        "model": target_model,
        "messages": message_list,
    }
    if reasoning_effort is None and "gpt-5" in target_model:
        payload["reasoning_effort"] = "minimal"
    elif reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    # Skip sampling controls for vision or reasoning models that require defaults.
    if not has_image and _allows_sampling_params(target_model):
        payload.update(
            {
                "temperature": DEFAULT_TEMPERATURE if temperature is None else temperature,
                "top_p": DEFAULT_TOP_P if top_p is None else top_p,
                "frequency_penalty": DEFAULT_FREQUENCY_PENALTY if frequency_penalty is None else frequency_penalty,
                "presence_penalty": DEFAULT_PRESENCE_PENALTY if presence_penalty is None else presence_penalty,
            }
        )

    num_attempts = 0
    while True:
        if num_attempts >= MAX_RETRIES:
            raise RuntimeError("OpenAI request failed after retries.")
        try:
            response = client.chat.completions.create(**payload)
            return response.choices[0].message.content.strip()
        except openai.AuthenticationError as exc:
            print(f"Authentication error: {exc}", file=sys.stderr)
            raise
        except openai.RateLimitError as exc:
            print(f"Rate limit error: {exc}", file=sys.stderr)
            print(f"Sleeping for {RETRY_DELAY_SECONDS}s...", file=sys.stderr)
            time.sleep(RETRY_DELAY_SECONDS)
            num_attempts += 1
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Request failed: {exc}", file=sys.stderr)
            print(f"Sleeping for {RETRY_DELAY_SECONDS}s...", file=sys.stderr)
            time.sleep(RETRY_DELAY_SECONDS)
            num_attempts += 1


def main() -> None:
    model_type = "gpt-5"
    sample_messages = [
        {"role": "user", "content": "Say hello and mention Azure OpenAI."},
    ]
    try:
        response_text = call_chat_completion(model_type, sample_messages)
    except MissingConfiguration as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(2)
    print(response_text)


if __name__ == "__main__":
    main()
