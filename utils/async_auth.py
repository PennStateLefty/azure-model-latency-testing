import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


load_dotenv()


def get_async_openai_client(endpoint: str | None = None):
    """Return an AsyncAzureOpenAI client for an Azure OpenAI / Foundry endpoint.

    Async counterpart to get_openai_client(). Works for both Chat Completions
    (client.chat.completions) and Responses (client.responses) APIs.
    Uses DefaultAzureCredential for auth.
    """
    from openai import AsyncAzureOpenAI

    endpoint = endpoint or os.getenv("AZURE_ENDPOINT")
    if not endpoint:
        raise ValueError(
            "No endpoint provided. Pass an endpoint or set AZURE_ENDPOINT in .env"
        )

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )

    return AsyncAzureOpenAI(
        azure_ad_token_provider=token_provider,
        api_version="2025-04-01-preview",
        azure_endpoint=endpoint,
    )
