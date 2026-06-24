# Module 15: Multimodal LLMs

## 🎯 Learning Objectives
- Use vision APIs to analyze images with LLMs
- Generate images with DALL-E and Stable Diffusion
- Transcribe audio with Whisper
- Build multimodal RAG systems that combine text and images
- Understand when multimodal is worth the extra cost

## 📚 What is Multimodal?

Most of this course uses text-in, text-out. Multimodal LLMs extend this to:
- **Vision**: Image + text → text (describe, analyze, OCR, extract data)
- **Generation**: Text → image (DALL-E, Stable Diffusion, Flux)
- **Audio**: Audio → text (Whisper transcription), Text → audio (TTS)
- **Multimodal RAG**: Retrieve images + text, reason over both

## 1. Vision APIs — Analyzing Images

### OpenAI GPT-4V / GPT-4o

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

### Analyzing local images

```python
import base64
from pathlib import Path

def encode_image(path: str) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract all text from this screenshot."},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image('screenshot.png')}"
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
    model="claude-sonnet-4-20250514",
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

### Use cases for vision
- **OCR**: Extract text from screenshots, documents, receipts
- **Document understanding**: Parse tables, forms, invoices
- **Accessibility**: Describe images for visually impaired users
- **Code from screenshots**: Convert UI mockups to HTML/CSS
- **Medical imaging**: Analyze X-rays, MRIs (with appropriate models)

## 2. Image Generation

### DALL-E 3

```python
response = client.images.generate(
    model="dall-e-3",
    prompt="A serene Japanese garden with a wooden bridge over a koi pond, watercolor style",
    size="1024x1024",
    quality="standard",
    n=1,
)

image_url = response.data[0].url
# Save locally
import httpx
httpx.get(image_url).content
```

### Prompt engineering for images
- Be specific about style: "watercolor", "photorealistic", "pixel art", "oil painting"
- Describe composition: "close-up", "wide angle", "bird's eye view"
- Include lighting: "golden hour", "dramatic lighting", "soft diffused light"
- Reference artists: "in the style of Monet", "Studio Ghibli aesthetic"

## 3. Audio — Whisper Transcription

```python
from pathlib import Path

audio_file = Path("recording.mp3")
with open(audio_file, "rb") as f:
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

### Text-to-Speech

```python
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Welcome to the LLM Engineering course!",
)
response.stream_to_file("output.mp3")
```

## 4. Multimodal RAG

Traditional RAG retrieves text chunks. Multimodal RAG adds image retrieval:

```
User Query
    │
    ▼
[Query Understanding] ──→ Text query? Image query? Both?
    │
    ├─→ [Text Embedding] ──→ Text Vector DB ──→ Text chunks
    │
    └─→ [CLIP Embedding] ──→ Image Vector DB ──→ Image chunks
    │
    ▼
[Context Assembly] ──→ Combine text + image context
    │
    ▼
[Multimodal LLM] ──→ GPT-4V / Claude with both text and images in context
```

### CLIP for image embeddings

```python
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image(image_path: str) -> list[float]:
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features[0].tolist()

def embed_text(text: str) -> list[float]:
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return features[0].tolist()
```

## 5. When to Use Multimodal

| Scenario | Worth it? | Why |
|----------|-----------|-----|
| Extracting text from PDFs/screenshots | Yes | Vision APIs are excellent at OCR |
| Describing products from photos | Yes | More accurate than manual tagging |
| Analyzing charts/graphs | Yes | LLMs can read visual data |
| Generating marketing images | Maybe | DALL-E quality varies; human review needed |
| Real-time video analysis | Expensive | Cost scales with frame count |

## 🧪 Hands-On Exercises

1. **Screenshot to Code**: Take a UI screenshot, send it to GPT-4V, and ask it to generate HTML/CSS. How close is the output?

2. **Receipt Parser**: Build a function that takes a photo of a receipt and extracts: store name, date, items, total. Test on 3 different receipts.

3. **Image RAG**: Build a small image library (10 images with captions). Use CLIP to embed both images and text queries. Retrieve the most relevant image for a text query.

4. **Cost Comparison**: Run the same analysis task with text-only description vs. vision API. Which produces better results and at what cost difference?

5. **Multimodal Pipeline**: Build a pipeline that: (1) transcribes an audio file with Whisper, (2) generates a summary, (3) creates an image illustration with DALL-E based on the summary.

## 🔗 Integration with Other Modules

- **Module 02 (RAG)**: Extend text RAG with image retrieval using CLIP
- **Module 06 (Optimization)**: Vision API calls are expensive — cache image analysis results
- **Module 10 (Guardrails)**: Validate image content before sending to vision APIs
- **Module 12 (Context)**: Images consume ~85 tokens each in GPT-4V — budget accordingly

## 📚 References

- [OpenAI Vision Guide](https://platform.openai.com/docs/guides/vision)
- [Anthropic Vision Docs](https://docs.anthropic.com/en/docs/build-with-claude/vision)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Whisper Paper](https://arxiv.org/abs/2212.04356)

---

**Multimodal is the future — every LLM app will eventually handle more than just text.**
