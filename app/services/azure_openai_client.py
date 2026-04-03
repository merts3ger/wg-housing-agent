import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def is_configured() -> bool:
    return bool(
        os.getenv("AZURE_OPENAI_ENDPOINT")
        and os.getenv("AZURE_OPENAI_API_KEY")
        and os.getenv("AZURE_OPENAI_DEPLOYMENT")
    )


def get_client() -> OpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    if not endpoint or not api_key:
        raise RuntimeError(
            "Azure OpenAI is not configured. "
            "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in your .env file."
        )

    endpoint = endpoint.rstrip("/")

    return OpenAI(
        api_key=api_key,
        base_url=f"{endpoint}/openai/v1/",
    )


def get_deployment() -> str:
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not deployment:
        raise RuntimeError(
            "AZURE_OPENAI_DEPLOYMENT is not set. "
            "Set it to your deployed model name in your .env file."
        )
    return deployment