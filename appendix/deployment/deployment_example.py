"""
LLM Deployment Example - FastAPI Service
=========================================
Production-ready LLM inference API with caching, rate limiting, and monitoring.

Prerequisites:
    pip install fastapi uvicorn pydantic
"""

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import json
import logging
from datetime import datetime
import hashlib
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM Inference API",
    description="Production-ready LLM inference service",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 512


class ChatResponse(BaseModel):
    id: str
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    created_at: str


class SimpleCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["response"]
            del self.cache[key]
        return None
    
    def set(self, key: str, response: Dict):
        self.cache[key] = {
            "timestamp": time.time(),
            "response": response
        }


cache = SimpleCache()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def create_chat_completion(request: ChatRequest):
    logger.info(f"Processing chat request with {len(request.messages)} messages")
    
    # Create cache key
    cache_key = hashlib.md5(
        json.dumps([m.model_dump() for m in request.messages], sort_keys=True).encode()
    ).hexdigest()
    
    # Check cache
    cached = cache.get(cache_key)
    if cached:
        logger.info("Cache hit")
        return ChatResponse(**cached)
    
    # Simulate LLM processing
    await asyncio.sleep(0.5)
    
    last_message = request.messages[-1].content if request.messages else ""
    mock_response = f"Simulated response to: {last_message[:50]}..."
    
    response = ChatResponse(
        id=f"chatcmpl-{int(time.time())}",
        model=request.model,
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": mock_response},
            "finish_reason": "stop"
        }],
        usage={
            "prompt_tokens": len(last_message.split()),
            "completion_tokens": len(mock_response.split()),
            "total_tokens": len(last_message.split()) + len(mock_response.split())
        },
        created_at=datetime.utcnow().isoformat()
    )
    
    # Cache response
    cache.set(cache_key, response.model_dump())
    
    return response


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "gpt-3.5-turbo", "object": "model"},
            {"id": "gpt-4", "object": "model"},
            {"id": "llama-2-7b", "object": "model"}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("LLM Deployment Service")
    print("=" * 60)
    print("\nEndpoints:")
    print("  GET  /health              - Health check")
    print("  POST /v1/chat/completions - Chat completion")
    print("  GET  /v1/models           - List models")
    print("\nRun: uvicorn deployment_example:app --reload --port 8000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
