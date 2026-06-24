"""Multimodal examples — vision, image generation, audio transcription."""

import os, sys, base64, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from openai import OpenAI

client = OpenAI()

G, Y, C, R, B, DIM = "\033[92m", "\033[93m", "\033[96m", "\033[91m", "\033[94m", "\033[90m"
RESET = "\033[0m"


def step(n, label):
    print(f"\n{G}▸ Step {n}:{RESET} {C}{label}{RESET}")


# ── 1. Vision — Analyze an image from URL ─────────────────────────────────────
step(1, "Vision — Image Analysis")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in 2 sentences. What is the main subject?"},
            {"type": "image_url", "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
            }}
        ]
    }],
    max_tokens=150,
)
print(f"   {response.choices[0].message.content}")


# ── 2. Vision — OCR (extract text from image) ────────────────────────────────
step(2, "Vision — OCR / Text Extraction")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract all visible text from this image. Return it as plain text."},
            {"type": "image_url", "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
            }}
        ]
    }],
    max_tokens=200,
)
print(f"   {response.choices[0].message.content}")


# ── 3. Image Generation ──────────────────────────────────────────────────────
step(3, "Image Generation — DALL-E 3")

response = client.images.generate(
    model="dall-e-3",
    prompt="A minimalist diagram showing a RAG pipeline: document → embeddings → vector DB → retrieval → LLM → answer. Clean, technical style on white background.",
    size="1024x1024",
    quality="standard",
    n=1,
)
image_url = response.data[0].url
print(f"   Generated image URL: {image_url[:80]}...")
print(f"   Revised prompt: {response.data[0].revised_prompt[:100]}...")


# ── 4. Audio Transcription ───────────────────────────────────────────────────
step(4, "Audio — Whisper Transcription (skip if no audio file)")

# Create a small test audio file using TTS, then transcribe it
print("   Creating test audio with TTS...")
tts_response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Large language models have revolutionized natural language processing. They can understand context, generate human-like text, and reason about complex problems.",
)
audio_path = Path("test_audio.mp3")
tts_response.stream_to_file(audio_path)
print(f"   {G}✓{RESET} Audio saved to {audio_path}")

# Transcribe it back
with open(audio_path, "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        response_format="verbose_json",
    )

print(f"   Transcript: {transcript.text}")
print(f"   Language:   {transcript.language}")
print(f"   Duration:   {transcript.duration:.1f}s")

# Cleanup
audio_path.unlink()


# ── 5. Multimodal Pipeline — Image description → Summary ─────────────────────
step(5, "Multimodal Pipeline — Image → Description → Summary")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": """Analyze this image and produce a structured JSON report:
{
  "main_subject": "...",
  "style": "...",
  "colors": ["..."],
  "suggested_caption": "...",
  "use_case": "what this image would be good for"
}"""},
            {"type": "image_url", "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
            }}
        ]
    }],
    response_format={"type": "json_object"},
    max_tokens=300,
)

report = json.loads(response.choices[0].message.content)
print(f"   Subject: {report.get('main_subject', 'N/A')}")
print(f"   Style:   {report.get('style', 'N/A')}")
print(f"   Colors:  {report.get('colors', [])}")
print(f"   Caption: {report.get('suggested_caption', 'N/A')}")
print(f"   Use:     {report.get('use_case', 'N/A')}")


# ── Done ──────────────────────────────────────────────────────────────────────
print(f"\n{G}{'='*60}")
print(f"  ✓ Multimodal demo complete!")
print(f"{'='*60}{RESET}\n")
print(f"  {DIM}What you saw:{RESET}")
print(f"  1. Vision API: analyzed images from URLs")
print(f"  2. OCR: extracted text from images")
print(f"  3. Image generation: created a diagram with DALL-E 3")
print(f"  4. Audio: TTS + transcription round-trip with Whisper")
print(f"  5. Pipeline: image → structured JSON report\n")
