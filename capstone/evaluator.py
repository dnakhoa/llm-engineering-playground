"""Async LLM-as-judge evaluation — Module 04 applied."""
import json
from openai import AsyncOpenAI

aclient = AsyncOpenAI()


async def evaluate_response(question: str, answer: str, context: str) -> dict:
    """Score answer quality asynchronously (fire-and-forget friendly)."""
    prompt = f"""Rate this RAG answer on 3 dimensions (1-5 each). Return JSON only.

Question: {question}
Context (retrieved): {context[:600]}
Answer: {answer}

{{
  "faithfulness": <1-5>,      // Is the answer grounded in the context?
  "relevance": <1-5>,         // Does it answer the question?
  "completeness": <1-5>,      // Is the answer sufficiently complete?
  "overall": <1-5>
}}"""

    try:
        resp = await aclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"faithfulness": 0, "relevance": 0, "completeness": 0, "overall": 0}
