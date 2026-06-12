# Environment Setup Guide

Get the playground running in 5 minutes.

---

## 1. Python Version

Python 3.10 or 3.11 recommended. Check yours:

```bash
python --version
```

---

## 2. Virtual Environment (recommended)

```bash
# Create
python -m venv .venv

# Activate — macOS/Linux
source .venv/bin/activate

# Activate — Windows
.venv\Scripts\activate
```

---

## 3. Install Dependencies

**All modules at once (easiest):**
```bash
pip install -r requirements.txt
```

**One module at a time (faster, less disk space):**
```bash
cd 01-prompt-engineering
pip install -r requirements.txt
```

> **Mac note:** `bitsandbytes` (GPU quantization, Module 03) is Linux/Windows only.
> Comment it out in `03-fine-tuning/requirements.txt` if on macOS.

---

## 4. API Keys

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

Then open `.env` and add at minimum:

```
OPENAI_API_KEY=sk-...
```

**Where to get API keys:**

| Service | URL | Cost |
|---------|-----|------|
| OpenAI | https://platform.openai.com/api-keys | Pay-as-you-go |
| Anthropic | https://console.anthropic.com/ | Pay-as-you-go |
| Hugging Face | https://huggingface.co/settings/tokens | Free tier available |
| Langfuse (Module 08) | https://cloud.langfuse.com | Free tier available |

> **Budget tip:** Modules 1–6 cost less than $1 total using `gpt-4o-mini`.
> Fine-tuning (Module 03) requires GPU access — use Google Colab or Kaggle if you don't have local GPU.

---

## 5. Verify Setup

```bash
python -c "
import openai, dotenv, langchain
dotenv.load_dotenv()
import os
key = os.getenv('OPENAI_API_KEY')
print('✅ Setup OK' if key and key.startswith('sk-') else '❌ Missing OPENAI_API_KEY in .env')
"
```

---

## 6. Run the Interactive Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or JupyterLab
jupyter lab
```

Then navigate to any module and open the `.ipynb` file.

**Recommended order:**
1. `01-prompt-engineering/prompt_engineering.ipynb`
2. `02-rag-systems/rag_systems.ipynb`
3. Read the README + run the `.py` script for modules 03–11

---

## 7. Troubleshooting

**`ModuleNotFoundError: langchain_community`**
```bash
pip install langchain-community
```

**`chromadb` import fails on Apple Silicon**
```bash
pip install chromadb --no-binary chromadb
```

**`bitsandbytes` fails on macOS**
```bash
# Comment out bitsandbytes in 03-fine-tuning/requirements.txt
# Run QLoRA in Google Colab instead (free GPU)
```

**OpenAI `AuthenticationError`**
- Check your `.env` file exists at the repo root (not inside a module folder)
- Verify the key starts with `sk-` and has no trailing spaces
