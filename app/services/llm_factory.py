"""
app/services/llm_factory.py
----------------------------
LLM Factory + Strategy Pattern.

Add a new provider by:
  1. Adding it to PROVIDER_REGISTRY
  2. Adding one 'if' branch in LLMFactory.create()
  No other files need to change.
"""

from typing import Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq

from app.core.config import settings


# ── Provider Registry ─────────────────────────────────────────────────────────
# Single source of truth for all supported providers and models.

PROVIDER_REGISTRY: Dict[str, List[str]] = {
    "groq": [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
    ],
    "ollama": [
        "llama3",
        "mistral",
        "phi3",
        "codellama",
    ],
}


# ── Factory ───────────────────────────────────────────────────────────────────

class LLMFactory:
    """
    Instantiates a LangChain BaseChatModel for any supported provider/model.
    All returned objects are directly usable in LangChain chains (|).
    """

    @staticmethod
    def create(
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> BaseChatModel:
        """
        Create and return a configured LLM instance.

        Args:
            provider: groq | openai | ollama  (defaults to settings.LLM_PROVIDER)
            model:    Model ID string          (defaults to settings.LLM_MODEL)
        """
        _provider = (provider or settings.LLM_PROVIDER).lower()
        _model    = model or settings.LLM_MODEL

        print(f"[LLMFactory] provider={_provider!r}  model={_model!r}")

        if _provider == "groq":
            return ChatGroq(
                model=_model,
                temperature=0,
                api_key=settings.GROQ_API_KEY,
            )

        if _provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as e:
                raise ImportError("pip install langchain-openai") from e
            return ChatOpenAI(
                model=_model,
                temperature=0,
                api_key=settings.OPENAI_API_KEY,
            )

        if _provider == "ollama":
            try:
                from langchain_ollama import ChatOllama
            except ImportError as e:
                raise ImportError("pip install langchain-ollama") from e
            return ChatOllama(
                model=_model,
                base_url=settings.OLLAMA_BASE_URL,
            )

        raise ValueError(
            f"Unknown provider: {_provider!r}. "
            f"Supported: {list(PROVIDER_REGISTRY.keys())}"
        )

    @classmethod
    def get_providers(cls) -> List[Dict[str, object]]:
        """Return provider/model registry as a list. Used by GET /llm/providers."""
        return [
            {"provider": p, "models": m}
            for p, m in PROVIDER_REGISTRY.items()
        ]

    @classmethod
    def validate(cls, provider: str, model: str) -> bool:
        return provider in PROVIDER_REGISTRY and model in PROVIDER_REGISTRY[provider]
