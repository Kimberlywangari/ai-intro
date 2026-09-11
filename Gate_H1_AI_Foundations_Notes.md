# Gate H1 — AI Landscape, LLMs & Tooling

*Training notes — Kimberly Njoroge*

## 1. Gate Objective & Requirements

Build a practical understanding of the modern AI landscape and establish a working environment for experimenting with LLMs and AI tools.

**Task:** Research and categorize the major AI subfields (ML, DL, NLP, CV, GenAI), identifying real-world applications, named examples, and how each relates to the broader AI field. Study the core concepts behind transformer-based LLMs (tokenization, embeddings, attention, context windows, transformers). Set up and verify a working AI development environment: Python, Jupyter, Hugging Face access, Kaggle access, and at least one LLM API (OpenAI or Gemini).

## 2. AI Subfields — How They Relate

The subfields nest as concentric circles, not separate boxes:

```
Artificial Intelligence (AI)
  └── Machine Learning (ML)
        └── Deep Learning (DL)
              └── Natural Language Processing (NLP)  ← modern NLP is mostly DL now
              └── Computer Vision (CV)                ← modern CV is mostly DL now
              └── Generative AI (GenAI)                ← a DL capability, cuts across NLP + CV
```

**Machine Learning (ML)**
The parent discipline: instead of hand-coding rules, a system learns patterns/statistical relationships from data. Ranges from simple linear regression to random forests to neural networks.
> *Example:* A bank's loan default predictor trained on historical applicant data (logistic regression or gradient boosting, e.g. XGBoost) — "classic ML," not deep learning.

**Deep Learning (DL)**
A subset of ML using artificial neural networks with many layers. Scales well with large datasets and compute — what unlocked the current AI boom.
> *Example:* ResNet — a deep convolutional neural network architecture used for image classification.

**Natural Language Processing (NLP)**
Machines understanding/generating human language. Historically rule-based/statistical; now dominated by deep learning (transformers).
> *Example:* spaCy — an NLP library used for tasks like extracting diagnoses or drug names from clinical notes .

**Computer Vision (CV)**
Machines interpreting visual data (images, video).
> *Example:* ResNet-driven satellite imagery analysis — a deep convolutional neural network used to detect deforestation or flood damage from satellite photos.

**Generative AI (GenAI)**
Not a separate pipeline but a capability: models that generate new content (text, images, audio) rather than just classifying/predicting. Overlaps heavily with NLP and CV, powered by deep learning.
> *Example:* GPT-4 (text generation, NLP+GenAI) and DALL·E (image generation, CV+GenAI).

> **One-line summary:** AI is the umbrella goal — making machines act intelligently. ML is the dominant approach — learning from data instead of hard-coded rules. DL is the ML technique using deep neural networks that now powers most of NLP and CV. GenAI isn't a separate field — it's what happens when those DL/NLP/CV models are used to produce new content instead of just labeling or predicting.

---

## 3. Core Concepts Behind Transformer-Based LLMs

Built in the order they occur when a prompt is processed:

**1. Tokenization**
Text is chopped into tokens — not quite words, not quite characters (e.g. "pneumothorax" → ["pneum", "othorax"]), using a tokenizer such as BPE (Byte Pair Encoding). Common words are often one token; rare/technical words get split. This is why LLMs are priced per token and why a "128k context window" is measured in tokens, not words.

**2. Embeddings**
Each token becomes a vector — a long list of numbers (e.g. 768 or 4096 dimensions) representing its meaning as a position in high-dimensional space. Words with similar meaning sit close togetheWords with similar meaning sit close together (e.g. "automobile," "car," and "vehicle" cluster near each other; "carpet" does not, despite sharing letters with "car").

**3. Attention**
From the 2017 paper "Attention Is All You Need." For every token, attention asks which other tokens in the sequence it should weight to be understood correctly. Example: in "she picked up the umbrella and opened it before stepping outside," attention lets the model resolve "it" to "umbrella" — a word several tokens earlier — by weighting that token heavily. Self-attention does this for every token against every other token simultaneously — enabling long-range context handling, unlike older word-by-word RNN models.

**4. Transformers**
The architecture packaging tokenization → embeddings → (multiple layers of) self-attention → output, trained end-to-end. "Transformer" is the blueprint; GPT, Gemini, Claude, and LLaMA are specific models built on it.

**5. Context Window**
The maximum number of tokens a model can "see" at once (input + output combined). If prompt + history exceeds it, older content is dropped or the request fails — why long documents are sometimes chunked/summarized before being fed to an LLM.

> **Chain, in one sentence:** Text → tokenized into pieces → each piece embedded as a vector → self-attention lets every token weigh every other token to build contextual meaning → this happens across many transformer layers → all within the limit of the context window.

---

## 4. AI Development Environment Setup

Sequence used (Windows laptop). Each step builds on the last.

**Step 1 — Install Python**
Download Python 3.11+ from python.org (not the Microsoft Store version — path issues). Tick "Add python.exe to PATH" during install. Verify with `python --version`.

**Step 2 — Install Jupyter Notebook**
`pip install notebook`, then run `jupyter notebook` — opens a browser tab at localhost:8888 showing the file system.

**Step 3 — Create a Hugging Face account**
huggingface.co → Settings → Access Tokens → create a Read token. Hugging Face is the "GitHub of ML models" — hosts pre-trained models, datasets, and the `transformers` library for running models locally rather than via a paid API. Represents one of two AI-usage paradigms: calling a hosted API (Gemini) versus downloading and running an open-source model's own weights.

**Step 4 — Create a Kaggle account**
kaggle.com → Settings → API → Create New Token, downloading `kaggle.json`. Kaggle is best known for ML competitions but matters here for its public dataset repository and free-tier GPU/TPU notebooks. Pairing: Hugging Face = pre-trained models, Kaggle = datasets (and cloud compute if a local machine can't handle heavier training).

**Step 5 — Get a Gemini API key**
aistudio.google.com/apikey → sign in → Create API key. Gemini's free tier was chosen over OpenAI to avoid a billing requirement while learning. The key is copied to a temporary safe note — not pasted into any code yet.

**Step 6 — Store the key as a Windows environment variable**
```
setx GEMINI_API_KEY "paste-your-actual-key-here"
```
Close and reopen PowerShell afterward (`setx` affects future sessions only). Verify with `echo $env:GEMINI_API_KEY`.

*Why:* hardcoding a key into a `.py`/`.ipynb` file risks it entering Git history permanently (retrievable even after deletion) and exposes it to anyone who opens the notebook. An OS-level environment variable keeps the value out of every tracked file — code only asks the OS whether the variable exists.

**Step 7 — Install the Gemini Python SDK**
```
pip install google-generativeai
```
(This package was later deprecated by Google in favor of `google-genai` — see Section 5.) An SDK wraps a service's raw HTTP API into normal Python objects/methods so the exact request URL, JSON shape, auth headers, error formats, and rate-limit handling don't need to be built by hand.

**Step 8 — Verify Jupyter can see the key**
```python
import os
print(bool(os.environ.get("GEMINI_API_KEY")))
```
Prints True/False rather than the raw key, so the credential itself is never displayed or saved into notebook output. If it prints False, Jupyter was likely already running before `setx` was used — restart it fully.

---

## 5. What an SDK Is (and why it matters beyond this gate)

SDK = Software Development Kit: a pre-built package wrapping a service's raw HTTP API so the underlying request/response plumbing doesn't need to be handled manually. Underneath, an API call is just an HTTP POST with a JSON payload to a URL, returning JSON — an SDK turns that into clean Python objects, handling auth, errors, retries, and streaming.

**google-genai (current SDK):** Google's official SDK for the Gemini API specifically (separate from the broader `google-cloud-*` packages). Provides a single `Client` object as the entry point for all capabilities — text generation, embeddings, file uploads, chat, structured JSON output, streaming, and multimodal (image + text) input.


---

