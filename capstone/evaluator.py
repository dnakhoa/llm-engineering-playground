"""Async LLM-as-judge evaluation — Module 04 applied, multi-provider aware."""
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.provider import _detect_provider, _default_model


async def evaluate_response(question: str, answer: str, context: str) -> dict:
    """Score answer quality asynchronously (fire-and-forget friendly)."""
    provider = _detect_provider()
    model = os.getenv("LLM_MODEL") or _default_model(provider)

    prompt = f"""Rate this RAG answer on 3 dimensions (1-5 each). Return JSON only.

Question: {question}
Context (retrieved): {context[:600]}
Answer: {answer}

{{
  "faithfulness": <1-5>,
  "relevance": <1-5>,
  "completeness": <1-5>,
  "overall": <1-5>
}}"""

    try:
        if provider == "anthropic":
            import anthropic
            client = anthropic.AsyncAnthropic()
            resp = await client.messages.create(
                model=model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return json.loads(resp.content[0].text)
        else:
            from openai import AsyncOpenAI
            kwargs = {"api_key": os.getenv("OPENAI_API_KEY") or os.getenv(f"{provider.upper()}_API_KEY", "no-key")}
            base_url = os.getenv("OPENAI_BASE_URL")
            if provider == "deepseek" and not base_url:
                base_url = "https://api.deepseek.com/v1"
            elif provider == "grok" and not base_url:
                base_url = "https://api.x.ai/v1"
            elif provider == "qwen" and not base_url:
                base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            if base_url:
                kwargs["base_url"] = base_url
            client = AsyncOpenAI(**kwargs)
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"faithfulness": 0, "relevance": 0, "completeness": 0, "overall": 0}
