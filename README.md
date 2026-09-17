# 🎬 AI Video Assistant

An end-to-end AI pipeline that takes a video or audio recording and turns it into a searchable, chattable knowledge base — transcription, summarization, action-item extraction, and a RAG-powered chat interface, all in one app.

**🔗 Live Demo:** [ai-meeting-video-assistant.streamlit.app](https://ai-meeting-video-assistant-m8zwcx6doemb25omlemvve.streamlit.app/)

---

## ✨ Features

- **Flexible input** — upload an audio/video file directly, try a bundled sample recording, or paste a YouTube URL (local runs)
- **Speech-to-text transcription** using OpenAI Whisper (default, runs locally with no rate limits) or Sarvam AI (`saaras:v2.5`) as an optional alternative, with automatic audio chunking for long recordings
- **AI-generated summaries** via an LLM, using map-reduce summarization for long transcripts
- **Automatic title generation** for each session
- **Structured extraction** of:
  - ✅ Action items (task, owner, deadline)
  - 🔑 Key decisions
  - ❓ Open/unresolved questions
- **RAG-powered chat** — ask natural-language questions about the transcript and get grounded, context-aware answers, backed by a local vector store (ChromaDB + HuggingFace embeddings)
- **Clean, custom-themed UI** built with Streamlit

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Groq (`openai/gpt-oss-20b`) via LangChain |
| Speech-to-Text | OpenAI Whisper / Sarvam AI (`saaras:v2.5`) |
| Embeddings | HuggingFace `sentence-transformers` (local, no API key required) |
| Vector Store | ChromaDB |
| Audio Processing | `pydub`, `ffmpeg` |
| YouTube Downloading | `yt-dlp` |
| Orchestration | LangChain (LCEL chains) |

---

## 📐 Architecture

```
Input (Upload / Sample / YouTube URL)
        │
        ▼
  Audio Processing (chunking, format conversion)
        │
        ▼
  Whisper Transcription
        │
        ├──► Title Generation (LLM)
        ├──► Summarization (map-reduce for long transcripts)
        ├──► Action Item / Decision / Question Extraction (LLM)
        └──► Vector Store (ChromaDB) ──► RAG Chat (retrieval + LLM)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- `ffmpeg` installed and on your PATH
- A [Groq API key](https://console.groq.com) (free tier available)

### Installation

```bash
git clone https://github.com/Azaanjafri313/AI-Meeting-Video-Assistant.git
cd AI-Meeting-Video-Assistant
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here   # optional — Sarvam's free tier has a low usage quota, so Whisper is used by default
```

### Run Locally

**Streamlit UI:**
```bash
streamlit run app.py
```

**CLI:**
```bash
python main.py
```

---

## 📥 Input Methods

The app supports three ways to provide a video/audio for analysis:

1. **📁 Upload File** *(recommended for the hosted demo)* — upload an audio or video file directly. Works reliably in any environment.
2. **🎬 Try Sample** — runs the full pipeline on a bundled demo recording, so you can see the app in action with zero setup.
3. **🔗 YouTube URL** — downloads and transcribes a YouTube video directly. This works reliably on local runs. **On cloud deployments, YouTube blocks download requests from datacenter IPs**, so this option is best-effort when hosted — use **Upload File** or **Try Sample** for a guaranteed demo experience.

> This is a known, documented constraint of YouTube's anti-scraping measures on cloud provider IP ranges — not a bug in this app.

---

## 📂 Project Structure

```
├── app.py                      # Streamlit UI entry point
├── main.py                     # CLI entry point
├── core/
│   ├── transcriber.py          # Whisper-based transcription
│   ├── summarizer.py           # Map-reduce summarization + title generation
│   ├── extractor.py            # Action items / decisions / questions extraction
│   ├── RAG_engine.py           # RAG chain for chat-with-transcript
│   ├── vector_store.py         # ChromaDB + HuggingFace embeddings
│   └── deno_setup.py           # Deno runtime setup for yt-dlp
├── utils/
│   └── audio_processor.py      # YouTube download, format conversion, chunking
├── assets/
│   └── sample_meeting.mp4      # Bundled demo file for "Try Sample"
└── requirements.txt
```

---

## ⚙️ Key Design Decisions

- **Map-reduce summarization** for transcripts longer than the model's comfortable context window, chunking with overlap to preserve continuity between segments.
- **Local embeddings (HuggingFace `all-MiniLM-L6-v2`)** instead of an API-based embedding model — keeps the RAG pipeline free and avoids an extra external dependency.
- **Graceful degradation** — title generation and per-chunk summarization failures don't crash the whole pipeline; they fall back gracefully and continue.
- **Multiple input paths** to keep the hosted demo reliable regardless of external platform restrictions (see [Input Methods](#-input-methods) above).

---

## ⚠️ Known Limitations

- **YouTube URL downloads** are blocked on cloud deployments due to YouTube restricting datacenter IPs — see [Input Methods](#-input-methods) above for the workaround (Upload File / Try Sample).
- **Sarvam AI transcription** has a low free-tier usage quota and can hit rate limits quickly under repeated use. **Whisper is used as the default and primary transcription engine** for this reason; Sarvam is available as an optional alternative for Hindi/regional-language accuracy but isn't the reliable default path.

---

## 🧗 Challenges & What I Learned

- Migrated from Mistral to Groq after hitting free-tier rate limits — learned to design around provider-specific constraints (rate limits, model deprecations) rather than hardcoding assumptions.
- Diagnosed and worked around YouTube's IP-based blocking of cloud/datacenter traffic, including experimenting with cookie-based authentication and ultimately designing a more robust multi-input UX instead of relying on a single fragile data source.
- Debugged subtle prompt-engineering bugs (a copy-pasted prompt causing the RAG chat to summarize instead of answering questions) — reinforced the importance of testing each chain in isolation.

---

## 📄 License

This project is for educational/portfolio purposes.
