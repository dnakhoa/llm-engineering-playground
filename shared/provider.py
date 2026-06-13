"""
Shared LLM Provider — works with OpenAI, Anthropic, and any OpenAI-compatible API.

Set one of these environment variables (or use .env):
    OPENAI_API_KEY=sk-...           → OpenAI
    ANTHROPIC_API_KEY=sk-ant-...    → Anthropic
    DEEPSEEK_API_KEY=sk-...         → DeepSeek (OpenAI-compatible)
    GROK_API_KEY=...                → xAI Grok (OpenAI-compatible)
    QWEN_API_KEY=...                → Alibaba Qwen (OpenAI-compatible)
    OPENAI_BASE_URL=http://localhost:11434/v1  → Ollama or any local server
    LLM_PROVIDER=openai|anthropic|deepseek|grok|qwen|ollama|custom
    LLM_MODEL=gpt-4o-mini           → override default model
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _detect_provider() -> str:
    """Auto-detect provider from environment variables."""
    explicit = os.getenv("LLM_PROVIDER", "").lower()
    if explicit:
        return explicit

    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("DEEPSEEK_API_KEY"):
        return "deepseek"
    if os.getenv("GROK_API_KEY"):
        return "grok"
    if os.getenv("QWEN_API_KEY"):
        return "qwen"
    if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL"):
        return "openai"
    return "openai"


def _default_model(provider: str) -> str:
    defaults = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-sonnet-4-20250514",
        "deepseek": "deepseek-chat",
        "grok": "grok-3-mini",
        "qwen": "qwen-plus",
        "ollama": "llama3.2",
    }
    return defaults.get(provider, "gpt-4o-mini")


def chat(
    prompt: str,
    *,
    system: str = None,
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    messages: list[dict] = None,
) -> str:
    """
    Send a prompt to the configured LLM and return the response text.

    Usage:
        from shared.provider import chat
        print(chat("What is 2+2?"))
        print(chat("Summarize this:", system="You are a helpful assistant"))
    """
    provider = _detect_provider()
    model = model or os.getenv("LLM_MODEL") or _default_model(provider)

    if messages is None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

    if provider == "anthropic":
        return _chat_anthropic(messages, model, temperature, max_tokens)
    else:
        return _chat_openai_compatible(messages, model, temperature, max_tokens, provider)


def _chat_openai_compatible(
    messages: list[dict], model: str, temperature: float, max_tokens: int, provider: str
) -> str:
    """Call any OpenAI-compatible API (OpenAI, DeepSeek, Grok, Qwen, Ollama)."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv(f"{provider.upper()}_API_KEY", "no-key")
    base_url = os.getenv("OPENAI_BASE_URL")

    if provider == "deepseek" and not base_url:
        base_url = "https://api.deepseek.com/v1"
    elif provider == "grok" and not base_url:
        base_url = "https://api.x.ai/v1"
    elif provider == "qwen" and not base_url:
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _chat_anthropic(
    messages: list[dict], model: str, temperature: float, max_tokens: int
) -> str:
    """Call the Anthropic API."""
    import anthropic

    client = anthropic.Anthropic()

    # Anthropic uses separate system parameter
    system_text = ""
    user_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"]
        else:
            user_messages.append(msg)

    kwargs = {"model": model, "messages": user_messages, "max_tokens": max_tokens, "temperature": temperature}
    if system_text:
        kwargs["system"] = system_text

    response = client.messages.create(**kwargs)
    return response.content[0].text


def get_client():
    """Return a raw client object for advanced usage."""
    provider = _detect_provider()
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic()
    else:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv(f"{provider.upper()}_API_KEY", "no-key")
        base_url = os.getenv("OPENAI_BASE_URL")

        if provider == "deepseek" and not base_url:
            base_url = "https://api.deepseek.com/v1"
        elif provider == "grok" and not base_url:
            base_url = "https://api.x.ai/v1"
        elif provider == "qwen" and not base_url:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)


def get_model_name() -> str:
    """Return the currently configured model name."""
    provider = _detect_provider()
    return os.getenv("LLM_MODEL") or _default_model(provider)
