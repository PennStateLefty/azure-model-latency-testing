import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.ai.inference import ChatCompletionsClient


load_dotenv()


def _normalize_openai_base_url(endpoint: str) -> str:
    """Normalize an endpoint into an OpenAI-compatible /openai/v1/ base URL."""
    normalized_endpoint = endpoint.rstrip("/")
    if normalized_endpoint.endswith("/openai/v1"):
        return f"{normalized_endpoint}/"
    if normalized_endpoint.endswith("/openai"):
        return f"{normalized_endpoint}/v1/"
    return f"{normalized_endpoint}/openai/v1/"


def get_client(endpoint: str | None = None) -> ChatCompletionsClient:
    """Return an authenticated ChatCompletionsClient for an Azure AI Foundry endpoint.

    Uses DefaultAzureCredential (managed identity in Azure, CLI creds locally).
    Falls back to AZURE_ENDPOINT env var if no endpoint is provided.
    """
    endpoint = endpoint or os.getenv("AZURE_ENDPOINT")
    if not endpoint:
        raise ValueError(
            "No endpoint provided. Pass an endpoint or set AZURE_ENDPOINT in .env"
        )

    credential = DefaultAzureCredential()
    return ChatCompletionsClient(endpoint=endpoint, credential=credential)


def get_openai_client(endpoint: str | None = None):
    """Return an AzureOpenAI client configured for Azure AI Foundry's Responses API.

    Uses DefaultAzureCredential (managed identity in Azure, CLI creds locally).
    The base_url is normalized to end with /openai/v1/.
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

    base_url = _normalize_openai_base_url(endpoint)
    return AzureOpenAI(
        azure_endpoint=endpoint.rstrip("/"),
        azure_ad_token_provider=token_provider,
        api_version="2025-04-01-preview",
        base_url=base_url,
    )


def get_model_name() -> str:
    """Return the model deployment name from environment."""
    name = os.getenv("AZURE_MODEL_NAME")
    if not name:
        raise ValueError("AZURE_MODEL_NAME not set in .env")
    return name
