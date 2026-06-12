"""Interactive terminal chat client for the Knowledge Assistant API."""
import httpx
import sys

BASE_URL = "http://localhost:8000"
SESSION_ID = "terminal-session"


def chat(question: str) -> dict:
    resp = httpx.post(
        f"{BASE_URL}/chat",
        json={"question": question, "session_id": SESSION_ID},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    print("Knowledge Assistant — LLM Engineering Playground Capstone")
    print("Type 'quit' to exit, 'stats' to see metrics\n")

    # Check server is running
    try:
        health = httpx.get(f"{BASE_URL}/health", timeout=3)
        print(f"Server: {health.json()['status']}\n")
    except Exception:
        print("❌ Server not running. Start it with: uvicorn app:app --port 8000")
        sys.exit(1)

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() == "quit":
            break
        if question.lower() == "stats":
            stats = httpx.get(f"{BASE_URL}/stats").json()
            print(f"Stats: {stats}\n")
            continue

        result = chat(question)
        cached_tag = " [CACHED]" if result["cache_hit"] else ""
        print(f"Assistant{cached_tag}: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])} | {result['latency_ms']}ms\n")


if __name__ == "__main__":
    main()
