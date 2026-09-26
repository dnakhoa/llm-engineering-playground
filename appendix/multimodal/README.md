# Module 15: Multimodal
> **Why this matters:** The world isn't text-only. Multimodal LLMs let you analyze images, generate visuals, transcribe audio, and build voice interfaces — expanding what LLM apps can do.

## Learning Objectives
- Use vision APIs for image analysis, OCR, and document understanding
- Generate images with DALL-E 3 and understand prompt engineering for images
- Transcribe audio with Whisper and build text-to-speech
- Build multimodal RAG systems that combine text and images
- Understand video generation and voice agents
- Know when multimodal is worth the extra cost

---

## What is Multimodal?

Most LLM apps use text-in, text-out. Multimodal LLMs extend this to:
- **Vision**: Image + text -> text (describe, analyze, OCR, extract data)
- **Generation**: Text -> image (DALL-E, Stable Diffusion, Flux)
- **Video**: Text -> video (OpenAI Sora, Google Veo)
- **Audio**: Audio -> text (Whisper), Text -> audio (TTS, Realtime API)
- **Multimodal RAG**: Retrieve images + text, reason over both

---

## 1. Vision APIs — Analyzing Images

### OpenAI GPT-4o Vision

```python
from openai import OpenAI

client = OpenAI()

# Analyze an image from URL
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image? Describe it in detail."},
            {"type": "image_url", "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
            }}
        ]
    }],
    max_tokens=500,
)
print(response.choices[0].message.content)
```

### Analyzing Local Images (Base64)

```python
import base64
from pathlib import Path

def encode_image(path):
    return base64.b64encode(Path(path).read_bytes()).decode()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract all text from this screenshot."},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image('screenshot.png')}",
                "detail": "high"  # "low" for faster/cheaper, "high" for dense images
            }}
        ]
    }],
)
```

### Claude Vision

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": encode_image("photo.jpg"),
            }},
            {"type": "text", "text": "What do you see?"}
        ]
    }]
)
```

### Vision Use Cases

| Use Case | Example Prompt | Best Model |
|----------|---------------|------------|
| **OCR** | "Extract all text from this image" | GPT-4o, Claude |
| **Document parsing** | "Extract the table as JSON" | GPT-4o |
| **UI analysis** | "List all buttons and their labels" | GPT-4o |
| **Chart reading** | "What are the values in this chart?" | GPT-4o, Claude |
| **Medical imaging** | "Describe any abnormalities" | Specialized models |
| **Code from screenshots** | "Generate HTML/CSS for this UI" | GPT-4o |

### Detail Parameter

| Detail | Tokens | Cost | Best For |
|--------|--------|------|----------|
| `auto` | Varies | Varies | Default, let model decide |
| `low` | ~85 | Cheap | Thumbnails, simple images |
| `high` | ~1105 | Expensive | Dense text, detailed analysis |

---

## 2. Image Generation

### DALL-E 3

```python
response = client.images.generate(
    model="dall-e-3",
    prompt="A serene Japanese garden with a wooden bridge over a koi pond, watercolor style",
    size="1024x1024",
    quality="standard",  # "standard" or "hd"
    n=1,
)

image_url = response.data[0].url
revised_prompt = response.data[0].revised_prompt  # DALL-E may revise your prompt
```

### Prompt Engineering for Images

| Element | Examples | Impact |
|---------|---------|--------|
| **Style** | "watercolor", "photorealistic", "pixel art", "oil painting" | Major |
| **Composition** | "close-up", "wide angle", "bird's eye view", "macro" | Major |
| **Lighting** | "golden hour", "dramatic lighting", "soft diffused" | Major |
| **Subject** | Be specific: "a tabby cat" not just "a cat" | Major |
| **Background** | "on a wooden table", "in a forest", "studio backdrop" | Medium |
| **Artist reference** | "in the style of Monet", "Studio Ghibli" | Medium |
| **Quality modifiers** | "highly detailed", "4K", "professional" | Minor |

### Image Generation Platforms

| Platform | Quality | Speed | Cost | Best For |
|----------|---------|-------|------|----------|
| DALL-E 3 | High | Medium | $0.04-0.08/image | General purpose |
| Stable Diffusion | High | Fast | Free (local) | Customization, fine-tuning |
| Flux | Very High | Medium | Varies | Photorealism |
| Midjourney | Very High | Medium | $10-60/mo | Artistic quality |

---

## 3. Video Generation

### OpenAI Sora / gpt-4o-video

```python
response = client.images.generate(
    model="gpt-4o-video",
    prompt="A timelapse of a flower blooming in a garden, cinematic lighting",
    size="1080p",
)

video_url = response.data[0].url
```

### Video Generation Best Practices

| Element | Examples |
|---------|---------|
| **Motion** | "camera slowly pans left", "person walks toward camera" |
| **Temporal** | "first X happens, then Y" |
| **Style** | "cinematic", "documentary", "animation", "stop-motion" |
| **Duration** | Be specific about length if possible |

### When to Use Video Generation

| Use Case | Worth It? | Why |
|----------|-----------|-----|
| Marketing/social media | Yes | Fast, cheap, no filming needed |
| Concept visualization | Yes | Show ideas without storyboarding |
| Training material | Maybe | Quality varies; review needed |
| Production film | No | Need human direction and control |

---

## 4. Audio — Whisper Transcription

### Basic Transcription

```python
from pathlib import Path

with open("recording.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        language="en",
        response_format="verbose_json",
    )

print(transcript.text)
print(f"Language: {transcript.language}")
print(f"Duration: {transcript.duration}s")
```

### Transcription Formats

| Format | Contents | Best For |
|--------|----------|----------|
| `text` | Just the text | Simple use cases |
| `verbose_json` | Text + timestamps + language | Subtitles, analysis |
| `srt` | Subtitle format | Video editing |
| `vtt` | Web subtitle format | Web players |

### Speaker Diarization

```python
# Whisper doesn't do speaker diarization natively
# Use pyannote.audio for speaker identification

from pyannote.audio import Pipeline

diarization = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = diarization("recording.mp3")

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"[{turn.start:.1f}s -> {turn.end:.1f}s] {speaker}")
```

### Text-to-Speech

```python
response = client.audio.speech.create(
    model="tts-1",  # or "tts-1-hd" for higher quality
    voice="alloy",  # alloy, echo, fable, onyx, nova, shimmer
    input="Welcome to the LLM Engineering course!",
    speed=1.0,  # 0.25 to 4.0
)
response.stream_to_file("output.mp3")
```

### Voice Comparison

| Voice | Style | Best For |
|-------|-------|----------|
| alloy | Neutral, balanced | General purpose |
| echo | Male, clear | Narration |
| fable | Expressive | Storytelling |
| onyx | Deep, authoritative | Professional |
| nova | Female, warm | Friendly assistant |
| shimmer | Soft, gentle | Meditation, calm |

---

## 5. Realtime Audio (Voice Agents)

OpenAI's Realtime API enables voice-based agent interactions with low latency:

```python
# Voice agents use WebSocket connection for real-time audio
# See OpenAI Realtime API docs for full implementation

# Key capabilities:
# - Sub-second voice-to-voice latency
# - Tool calling during voice conversations
# - Voice activity detection (VAD)
# - Interruption handling (user can interrupt)
# - Multiple voice options
```

### Voice Agent Architecture

```
Microphone -> VAD -> STT -> LLM -> TTS -> Speaker
                |                |
                +--- Tool Use ---+
                     (weather, search, etc.)
```

### When to Use Voice Agents

| Use Case | Worth It | Why |
|----------|----------|-----|
| Customer support | Yes | Natural, hands-free |
| Accessibility | Yes | Essential for visually impaired |
| Real-time translation | Yes | Low-latency requirement |
| Content creation | Maybe | TTS quality varies |
| Simple Q&A | No | Text is faster |

---

## 6. Multimodal RAG

Traditional RAG retrieves text chunks. Multimodal RAG adds image retrieval:

```
User Query
    |
    v
[Query Understanding] --> Text query? Image query? Both?
    |
    +--> [Text Embedding] --> Text Vector DB --> Text chunks
    |
    +--> [CLIP Embedding] --> Image Vector DB --> Image chunks
    |
    v
[Context Assembly] --> Combine text + image context
    |
    v
[Multimodal LLM] --> GPT-4o / Claude with both text and images
```

### CLIP for Image Embeddings

```python
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        return model.get_image_features(**inputs)[0].tolist()

def embed_text(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        return model.get_text_features(**inputs)[0].tolist()
```

### Multimodal RAG Workflow

```python
# 1. Index: store both text and images
def index_documents(docs_with_images):
    for doc, images in docs_with_images:
        # Store text embedding
        text_emb = embed_text(doc["text"])
        vector_db.add(text_emb, {"type": "text", "content": doc["text"]})

        # Store image embeddings
        for img_path in images:
            img_emb = embed_image(img_path)
            vector_db.add(img_emb, {"type": "image", "path": img_path})

# 2. Query: find relevant text AND images
def multimodal_search(query, top_k=3):
    query_emb = embed_text(query)
    results = vector_db.search(query_emb, top_k=top_k)
    return results

# 3. Generate: feed both to multimodal LLM
def multimodal_rag(query):
    results = multimodal_search(query)
    # Build context with text and image references
    context = build_context(results)
    return llm_with_vision(query, context)
```

---

## 7. Document AI — Parsing Complex Documents

### Table Extraction

```python
# Use vision to extract tables from images/PDFs
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract this table as JSON with column headers."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{table_image}"}}
        ]
    }],
    response_format={"type": "json_object"},
)
```

### PDF Parsing Pipeline

```
PDF -> Page Images -> Vision API -> Structured Data
         |
         +-> OCR (if text layer exists)
         |
         +-> Table Detection -> Table Extraction
         |
         +-> Chart Detection -> Chart Analysis
```

---

## 8. Cost Analysis

### Vision API Costs (Approximate)

| Model | Input Tokens | Output Tokens | Cost per 1K Images |
|-------|-------------|---------------|-------------------|
| GPT-4o (low detail) | ~85/image | ~100 | ~$0.15 |
| GPT-4o (high detail) | ~1105/image | ~100 | ~$2.00 |
| Claude Sonnet 5 | ~1500/image | ~100 | ~$3.00 |
| GPT-4o-mini | ~85/image | ~100 | ~$0.05 |

### Audio Costs

| Model | Cost |
|-------|------|
| Whisper-1 | $0.006/minute |
| TTS-1 | $0.015/1K chars |
| TTS-1-HD | $0.030/1K chars |

### Cost Optimization Tips

| Technique | Savings |
|-----------|---------|
| Use `detail: "low"` for simple images | 90% |
| Cache image analysis results | 50-80% |
| Batch similar images | 20-40% |
| Use smaller models for OCR | 70% |

---

## When to Use Multimodal

| Scenario | Worth it? | Why |
|----------|-----------|-----|
| Extracting text from PDFs/screenshots | Yes | Vision APIs excel at OCR |
| Describing products from photos | Yes | More accurate than manual tagging |
| Analyzing charts/graphs | Yes | LLMs can read visual data |
| Generating marketing images | Maybe | Quality varies; human review needed |
| Generating marketing videos | Emerging | Improving rapidly |
| Voice-based agent interfaces | Yes | Realtime API enables low-latency voice UX |
| Real-time video analysis | Expensive | Cost scales with frame count |
| Document parsing (tables, forms) | Yes | Much more reliable than rule-based OCR |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Vision API can't read text in image | Use detail: "high" for dense images |
| Image generation produces poor results | Be specific about style, composition, lighting |
| Whisper transcription is inaccurate | Specify language parameter; use verbose_json |
| Audio file too large | Compress with ffmpeg; split into chunks |
| CLIP retrieval returns wrong images | Try larger CLIP model; add metadata filtering |
| Video generation produces artifacts | Simplify prompt; describe motion clearly |
| Voice agent has high latency | Use streaming; reduce context length |

## Resources

- [OpenAI Vision Guide](https://platform.openai.com/docs/guides/vision) — image analysis
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — voice agents
- [Anthropic Vision Docs](https://docs.anthropic.com/en/docs/build-with-claude/vision) — Claude vision
- [CLIP Paper](https://arxiv.org/abs/2103.00020) — image-text embeddings
- [Whisper](https://github.com/openai/whisper) — speech recognition
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — speaker diarization

## Exercises

1. **Screenshot to Code**: Take a UI screenshot, send to GPT-4o, ask for HTML/CSS. How close is the output?
2. **Receipt Parser**: Build a function that takes a receipt photo and extracts: store, date, items, total.
3. **Image RAG**: Build a 10-image library with CLIP. Retrieve relevant images for text queries.
4. **Cost Comparison**: Run same analysis with text-only vs vision API. Compare quality and cost.
5. **Multimodal Pipeline**: (1) Transcribe audio with Whisper, (2) summarize, (3) generate illustration with DALL-E.
6. **Table Extraction**: Extract a table from a photo. Convert to pandas DataFrame.
7. **Voice Agent**: Build a simple voice agent that responds to weather queries.

---

**Multimodal is the future — every LLM app will eventually handle more than just text.**
