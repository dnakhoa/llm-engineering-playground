"""Multimodal examples — vision, image generation, audio transcription.

Images and audio are not part of the course's text-only provider layer (llm/), so
this page calls OpenAI's SDK directly: vision through the Responses API, which
current OpenAI models use, and the dedicated image and audio models.

Models (checked against OpenAI's deprecations page, 2026-09-26):
    VISION_MODEL   an OpenAI chat model from llm/models.json
    IMAGE_MODEL    gpt-image-2      (replaces dall-e-3, shut down 2026-05-12)
    TTS_MODEL      gpt-4o-mini-tts
    STT_MODEL      gpt-transcribe   (replaces whisper-1, shutting down 2027-02-26)
"""

import os, sys, base64, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from openai import OpenAI

client = OpenAI()

VISION_MODEL = "gpt-6-luna"
IMAGE_MODEL = "gpt-image-2"
TTS_MODEL = "gpt-4o-mini-tts"
STT_MODEL = "gpt-transcribe"

CAT_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
PNG_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

G, Y, C, R, B, DIM = "\033[92m", "\033[93m", "\033[96m", "\033[91m", "\033[94m", "\033[90m"
RESET = "\033[0m"


def step(n, label):
    print(f"\n{G}▸ Step {n}:{RESET} {C}{label}{RESET}")


def look(prompt, image_url, **kwargs):
    """One image plus a question, through the Responses API."""
    return client.responses.create(
        model=VISION_MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": image_url},
            ],
        }],
        **kwargs,
    )


# ── 1. Vision — Analyze an image from URL ─────────────────────────────────────
step(1, "Vision — Image Analysis")

response = look("Describe this image in 2 sentences. What is the main subject?", CAT_URL,
                max_output_tokens=150)
print(f"   {response.output_text}")


# ── 2. Vision — OCR (extract text from image) ────────────────────────────────
step(2, "Vision — OCR / Text Extraction")

response = look("Extract all visible text from this image. Return it as plain text.", PNG_URL,
                max_output_tokens=200)
print(f"   {response.output_text}")


# ── 3. Image Generation ──────────────────────────────────────────────────────
step(3, f"Image Generation — {IMAGE_MODEL}")

response = client.images.generate(
    model=IMAGE_MODEL,
    prompt="A minimalist diagram showing a RAG pipeline: document → embeddings → vector DB → retrieval → LLM → answer. Clean, technical style on white background.",
    size="1024x1024",
    n=1,
)
# GPT Image models return the image itself, base64-encoded, rather than a URL.
image_path = Path("rag_pipeline.png")
image_path.write_bytes(base64.b64decode(response.data[0].b64_json))
print(f"   {G}✓{RESET} Image saved to {image_path}")


# ── 4. Audio Transcription ───────────────────────────────────────────────────
step(4, "Audio — Text to Speech, then Transcription")

# Create a small test audio file using TTS, then transcribe it
print("   Creating test audio with TTS...")
audio_path = Path("test_audio.mp3")
with client.audio.speech.with_streaming_response.create(
    model=TTS_MODEL,
    voice="alloy",
    input="Large language models have revolutionized natural language processing. They can understand context, generate human-like text, and reason about complex problems.",
) as tts_response:
    tts_response.stream_to_file(audio_path)
print(f"   {G}✓{RESET} Audio saved to {audio_path}")

# Transcribe it back
with open(audio_path, "rb") as f:
    transcript = client.audio.transcriptions.create(model=STT_MODEL, file=f)

print(f"   Transcript: {transcript.text}")

# Cleanup
audio_path.unlink()


# ── 5. Multimodal Pipeline — Image description → Summary ─────────────────────
step(5, "Multimodal Pipeline — Image → Description → Summary")

response = look("""Analyze this image and produce a structured JSON report:
{
  "main_subject": "...",
  "style": "...",
  "colors": ["..."],
  "suggested_caption": "...",
  "use_case": "what this image would be good for"
}""", CAT_URL, text={"format": {"type": "json_object"}}, max_output_tokens=300)

report = json.loads(response.output_text)
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
print(f"  1. Vision: analyzed images from URLs through the Responses API")
print(f"  2. OCR: extracted text from images")
print(f"  3. Image generation: created a diagram with {IMAGE_MODEL}")
print(f"  4. Audio: text to speech, then transcription with {STT_MODEL}")
print(f"  5. Pipeline: image → structured JSON report\n")
