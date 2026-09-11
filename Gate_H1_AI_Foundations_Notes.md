# Gate H1 — AI Landscape, LLMs & Tooling

*Training notes — Kimberly Njoroge*

## 1. Gate Objective & Requirements

Build a practical understanding of the modern AI landscape and establish a working environment for experimenting with LLMs and AI tools.

**Task:** Research and categorize the major AI subfields (ML, DL, NLP, CV, GenAI), identifying real-world applications, named examples, and how each relates to the broader AI field. Study the core concepts behind transformer-based LLMs (tokenization, embeddings, attention, context windows, transformers). Set up and verify a working AI development environment: Python, Jupyter, Hugging Face access, Kaggle access, and at least one LLM API (OpenAI or Gemini).

**Required Deliverables**
- AI Landscape Map — subfields, relationships, ≥10 named AI technologies/models/tools with a short explanation of what each does and which subfield it belongs to.
- LLM API Notebook — loads the API key from an environment variable, sends a request, receives a response, produces and prints a structured (e.g. JSON) response, runs start to finish. API keys must never be hardcoded or committed to Git.

**Pass Criteria**
- Landscape map is accurate, organized, contains ≥10 named examples — explainable without reading from the submission.
- Can explain the difference between ML, DL, NLP, CV, and GenAI.
- Can explain tokenization, embeddings, attention, transformers, and context windows in own words.
- Python and Jupyter installed and working.
- Notebook successfully calls the chosen LLM API; credentials stored in environment variables, never hardcoded.
- Can explain the notebook code line by line, without prompting; no code committed that can't be explained.
- Required day-floor met; commit history and daily check-ins show steady, spread-out progress rather than a last-minute burst.

**Evidence to Submit**
- AI landscape map
- Completed Jupyter notebook
- Evidence of a successful LLM API call
- Git commit history
- Daily check-in records

---

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
> *Example:* A hospital readmission risk calculator trained on historical patient data (logistic regression or gradient boosting, e.g. XGBoost) — "classic ML," not deep learning.

**Deep Learning (DL)**
A subset of ML using artificial neural networks with many layers. Scales well with large datasets and compute — what unlocked the current AI boom.
> *Example:* ResNet — a deep convolutional neural network architecture used for image classification.

**Natural Language Processing (NLP)**
Machines understanding/generating human language. Historically rule-based/statistical; now dominated by deep learning (transformers).
> *Example:* spaCy — an NLP library used for tasks like extracting diagnoses or drug names from clinical notes (named entity recognition).

**Computer Vision (CV)**
Machines interpreting visual data (images, video).
> *Example:* CheXNet — a deep learning model trained to detect pneumonia from chest X-rays (Stanford).

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
Each token becomes a vector — a long list of numbers (e.g. 768 or 4096 dimensions) representing its meaning as a position in high-dimensional space. Words with similar meaning sit close together (e.g. "myocardial infarction," "heart attack," and "MI" cluster near each other; "mitral" does not necessarily).

**3. Attention**
From the 2017 paper "Attention Is All You Need." For every token, attention asks which other tokens in the sequence it should weight to be understood correctly. Example: in "the patient took it despite the nausea," attention lets the model resolve "it" to a drug named several sentences earlier by weighting that token heavily. Self-attention does this for every token against every other token simultaneously — enabling long-range context handling, unlike older word-by-word RNN models.

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

**Where this transfers beyond the gate:** the same pattern (API key from env var → official SDK → structured request/response) applies anywhere AI is bolted onto a real project — e.g. auto-generating a plain-English weekly summary from logged workout/nutrition data for the workout-nutrition project, parsing free-text meal entries ("2 eggs and toast") into structured DB fields instead of brittle regex, or later extracting structured fields from unstructured clinical notes (the same task shape as NLP entity extraction, via an API rather than a local model).

---

## 6. Git & GitHub Submission Workflow

**1. Initialize the repo**
```
git init
```

**2. Create `.gitignore` before the first commit**
```
.ipynb_checkpoints/
*.env
.env
kaggle.json
```
`.ipynb_checkpoints/` is a hidden folder Jupyter auto-generates holding backup copies of the notebook (can contain stale output). `kaggle.json` holds Kaggle credentials. The Gemini key is already safe (lives only in an environment variable, not a file), but this `.gitignore` is the safety net for other credentials created in later gates.

**3. Create a GitHub repo**
github.com → New repository → name e.g. `gate-h1-ai-foundations` → do not initialize with a README → Create.

**4. Connect and push**
```
git remote add origin https://github.com/your-username/gate-h1-ai-foundations.git
git branch -M main
```

**5. Commit in stages (the "spread-out progress" evidence)**
```
git add .gitignore
git commit -m "Add gitignore for Jupyter checkpoints and credentials"

git add gate_h1_llm_notebook.ipynb
git commit -m "Add notebook: env var key loading and basic Gemini request"

git commit -am "Add structured JSON response cell"

git push -u origin main
```
*Note:* since setup and the initial notebook build were completed in one sitting, commit timestamps will cluster on that day unless remaining work (landscape map, refinements, README) is genuinely spread across following days rather than finished in one sweep.

**6. Daily check-ins**
A separate requirement from the mentor (Slack message, standup doc, or similar) — Git commit timestamps alone don't count as check-ins; format should be confirmed directly with the mentor if not already specified.

---

## 7. Status & Outstanding Items

- Subfields hierarchy and core LLM concepts (tokenization, embeddings, attention, transformers, context windows) — covered, with the requirement to be able to restate each without notes.
- Dev environment (Python, Jupyter, Hugging Face, Kaggle, Gemini API key, env var storage, SDK) — fully set up and verified (env var check printed True).
- Git/GitHub submission workflow (init, .gitignore, staged commits, push) — outlined, ready to execute.
- AI Landscape Map (≥10 named technologies/tools, relationships between subfields, one-line explanation each) — not yet built; concepts and examples above (ResNet, spaCy, CheXNet, GPT-4, DALL·E, plus Hugging Face/Kaggle/Gemini API itself) are the raw material to turn into the actual map/diagram deliverable.
- Daily check-in format — not yet confirmed with mentor.
