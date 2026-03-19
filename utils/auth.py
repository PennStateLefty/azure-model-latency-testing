import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


load_dotenv()


def get_client(endpoint: str | None = None):
    """Return an AzureOpenAI client for an Azure OpenAI / Foundry endpoint.

    Works for both Chat Completions (client.chat.completions) and Responses
    (client.responses) APIs.  Uses DefaultAzureCredential for auth.
    Falls back to AZURE_ENDPOINT env var if no endpoint is provided.
    """
    return get_openai_client(endpoint=endpoint)


def get_openai_client(endpoint: str | None = None):
    """Return an AzureOpenAI client for an Azure OpenAI / Foundry endpoint.

    Works for both Chat Completions (client.chat.completions) and Responses
    (client.responses) APIs.  Uses DefaultAzureCredential for auth.
    Falls back to AZURE_ENDPOINT env var if no endpoint is provided.
    """
    from openai import AzureOpenAI

    endpoint = endpoint or os.getenv("AZURE_ENDPOINT")
    if not endpoint:
        raise ValueError(
            "No endpoint provided. Pass an endpoint or set AZURE_ENDPOINT in .env"
        )

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )

    return AzureOpenAI(
        azure_ad_token_provider=token_provider,
        api_version="2025-04-01-preview",
        azure_endpoint=endpoint,
    )


def get_model_name() -> str:
    """Return the model deployment name from environment."""
    name = os.getenv("AZURE_MODEL_NAME")
    if not name:
        raise ValueError("AZURE_MODEL_NAME not set in .env")
    return name
