# 🧠 Memory-Aware Chatbot with Streamlit UI

Week 2 — Generative AI & Prompt Engineering internship assignment.

- **Live app:** `<ADD-YOUR-STREAMLIT-APP-URL-HERE>` *(placeholder — not deployed yet)*
- **GitHub repository:** `<ADD-YOUR-GITHUB-REPO-URL-HERE>` *(placeholder)*

## Overview

A chat application that keeps a multi-turn conversation inside a configurable **token budget**
(default **4,000 tokens**). When the conversation grows too large, it **summarizes older turns**
into a memory summary instead of silently deleting them, then shows token usage and compression
events in a dashboard.

## Problem statement

LLMs have a limited context window and charge per token. Naively sending the whole history
overflows the limit; naively cutting old messages makes the bot forget important facts. This
project demonstrates a middle path: token-aware context management with summarization.

## Features

- Multi-turn chat using Streamlit session state, with distinct user/assistant bubbles
- Adjustable token budget (500–16,000, default 4,000) and "recent turns kept" setting
- Real tokenizer (`tiktoken`, `cl100k_base`) with a heuristic fallback; estimates are always labelled as estimates
- Automatic summarization of the oldest turns, merged with the existing memory summary
- Budget enforced **before** every request; oversized requests are blocked, never sent
- Summarization failure never discards messages (clear error, request blocked)
- Token dashboard: budget, system prompt, memory summary, recent history, total, remaining, progress bar, compression counters, API-reported usage of the last response (when provided)
- Expandable memory-summary panel; summarized messages stay visible in a collapsed "Earlier messages" panel
- Reset Conversation button with confirmation dialog
- Works with any OpenAI-compatible API; clearly labelled **offline demo mode** when no key is set
- 45 automated pytest tests with mocked providers (no key or internet needed)

## Technology stack

Python 3.10+ · Streamlit · tiktoken · openai SDK (OpenAI-compatible APIs) · python-dotenv · pytest

## Project architecture

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit UI, sidebar, dashboard, chat loop, reset dialog |
| `src/config.py` | Settings from Streamlit secrets / environment variables, defaults, budget limits |
| `src/token_counter.py` | tiktoken counting + fallback estimator |
| `src/conversation.py` | Transcript, summary, compression statistics, reset |
| `src/memory_manager.py` | Budget measurement, turn selection, summarization, overflow protection |
| `src/llm_provider.py` | `OpenAICompatibleProvider`, `OfflineProvider`, provider factory |

The same provider object produces chat replies **and** memory summaries. See
[`MEMORY_STRATEGY.md`](MEMORY_STRATEGY.md) for the full algorithm and flow diagram.

## Directory structure

```
memory_aware_chatbot/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── MEMORY_STRATEGY.md
├── LICENSE
├── pytest.ini
├── .streamlit/
│   └── config.toml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── token_counter.py
│   ├── conversation.py
│   ├── memory_manager.py
│   └── llm_provider.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_token_counter.py
    ├── test_conversation.py
    ├── test_memory_manager.py
    ├── test_config_and_provider.py
    └── test_app_smoke.py
```

## Prerequisites

- Python 3.10 or newer (check with `py --version`)
- Git (only needed for GitHub upload)
- Optional: an API key for an OpenAI-compatible provider. Without one the app runs in offline demo mode.

## Local setup (Windows PowerShell)

```powershell
# 1. Open the project folder
cd path\to\memory_aware_chatbot

# 2. Create and activate a virtual environment
py -m venv venv
.\venv\Scripts\Activate.ps1
# If PowerShell blocks the script, run this once, then activate again:
# Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app opens at <http://localhost:8501>.

## API-key configuration

```powershell
copy .env.example .env
notepad .env
```

Edit the values (the real `.env` is ignored by git and must never be uploaded):

```
OPENAI_API_KEY=sk-...your real key...
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

To use another OpenAI-compatible provider (OpenRouter, Groq, etc.), change `OPENAI_BASE_URL`
and `LLM_MODEL` accordingly. Optional settings: `TOKEN_ENCODING`, `TOKEN_BUDGET`, `OFFLINE_MODE`.

The sidebar shows whether the app is **Live** (real LLM) or in **Offline demo mode**.

## Offline / demo mode

With no `OPENAI_API_KEY` (or with `OFFLINE_MODE=true`) the app uses a rule-based stand-in.
**It is not an AI model**: replies are simple templated text (labelled "Offline demo mode"), and
summaries are built by regular-expression extraction. The token counting, budget enforcement,
compression and dashboard are the real implementation, so you can test everything without a key.
Try it: set the budget slider to 500 and chat for a few turns.

## Testing

```powershell
pytest
```

Expected: `45 passed`. Tests use mocked providers and the offline provider — no key or internet
needed. `tests/test_app_smoke.py` runs the real Streamlit script headlessly in offline mode.

## Example token-budget workflow

1. Start the app, set **Context token budget** to 500, tell the bot: *"My name is Sara. I prefer short answers."*
2. Keep chatting. The **Total input context** metric and progress bar grow.
3. When the total would exceed 500, the oldest turns are summarized: a blue banner reports the
   compression, **Memory summary** gains lines such as `- [Facts] My name is Sara.`, and the
   summarized messages move into the collapsed "Earlier messages" panel.
4. Ask *"What is my name?"* — the answer still works because the fact lives in the memory summary.
5. Click **Reset Conversation** and confirm: messages, summary and counters are cleared.

## Deploy on Streamlit Community Cloud

1. Push the project to GitHub (next section). Never commit `.env`.
2. Go to <https://share.streamlit.io>, sign in with GitHub, and click **Create app**.
3. Choose your repository, branch `main`, and main file path `app.py`.
4. Open **Advanced settings → Secrets** and paste (using your own key):
   ```toml
   OPENAI_API_KEY = "sk-...your real key..."
   OPENAI_BASE_URL = "https://api.openai.com/v1"
   LLM_MODEL = "gpt-4o-mini"
   ```
5. Click **Deploy**. Without secrets the deployed app still works in offline demo mode.
6. Copy the app URL into the placeholder at the top of this README.

## GitHub upload commands

Create an empty repository on <https://github.com/new> (no README), then:

```powershell
git init
git add .
git status        # confirm .env and venv are NOT listed
git commit -m "Week 2: Memory-aware chatbot with Streamlit UI"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `streamlit` not recognized | Activate the venv first, then `pip install -r requirements.txt` |
| Sidebar says "Offline demo mode (no API key)" | Create `.env` from `.env.example` and set `OPENAI_API_KEY`, then restart the app |
| Sidebar says tokenizer fallback in use | `tiktoken` needs internet once to download its vocabulary; the app still works with the heuristic estimator |
| "AI provider returned an error" | Check key, model name, base URL and network; your message is returned so you can resend it |
| "Request was NOT sent" budget message | Shorten the message or raise the budget slider |
| Chat disappears after refreshing the browser | Expected: state lives in session memory only |

## License

MIT — see [`LICENSE`](LICENSE).
