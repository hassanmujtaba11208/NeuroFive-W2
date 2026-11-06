# Memory Strategy

This document describes exactly how the Memory-Aware Chatbot manages conversation memory.
Everything below reflects what the code in `src/` actually does.

## 1. The problem: limited context windows

A language model only "remembers" what is sent to it in each request. That request (the
*context*) has a hard size limit measured in tokens, and larger contexts cost more money and
respond more slowly. A chat that keeps appending every message will eventually overflow the
limit or become needlessly expensive.

## 2. Why silently dropping old messages is bad

The simplest fix is to cut off the oldest messages. That makes the bot forget the user's name,
preferences, decisions and unfinished tasks without any warning, and the user cannot tell what
was forgotten. This project instead **summarizes** older turns into a compact memory and tells
the user when it happens. Old messages are never deleted from the transcript.

## 3. Default budget: 4,000 tokens

`DEFAULT_TOKEN_BUDGET = 4000` (`src/config.py`). It is a practical teaching size: small enough
that compression can be seen after a modest conversation, large enough to hold a useful summary
plus several recent turns, and well below the context window of common models. The sidebar slider
allows 500–16,000 tokens (step 100); `TOKEN_BUDGET` in `.env`/secrets sets the starting value.

**What the budget covers:** the *input* context only — system prompt + memory summary + recent
messages (+ small format overhead). The model's reply (limited to 600 output tokens by
`DEFAULT_MAX_RESPONSE_TOKENS`) is not part of the budget. Pick a budget comfortably below your
model's real context window so input + output fit.

## 4. Tokenizer and its limitations

`src/token_counter.py` uses **tiktoken** with the **`cl100k_base`** encoding (configurable via
`TOKEN_ENCODING`). tiktoken downloads its vocabulary file the first time an encoding is used, so
it needs internet access once. If tiktoken or the encoding cannot be loaded, a **heuristic
fallback** is used: roughly 4 ASCII characters per token, 1 token per non-ASCII character, and
never fewer tokens than words (this errs on the high, safe side). The sidebar always shows which
method is active.

Limitations: every locally computed number is an **estimate**. The real count depends on the
provider's model and tokenizer, the exact message formatting, and API overhead. The UI labels
local numbers as estimates and only calls a number "API-reported" when it comes from the
provider's own `usage` field.

## 5. How token usage is calculated

`MemoryManager.measure()` computes, for the context that *would* be sent:

| Component | How it is counted |
|---|---|
| System prompt | tokens of the prompt text + role + ~4 message-framing tokens |
| Memory summary | tokens of the summary block (header + summary text) |
| Recent history | for every *active* (not yet summarized) message: content + role + ~4 framing tokens |
| Overhead | ~3 tokens for reply priming |
| **Total input context** | sum of the above; **remaining** = budget − total |

The system prompt and the summary are sent together as **one** system message (some providers
dislike several), but they are measured separately for the dashboard.

## 6. How recent turns are retained

A *turn* is one user message plus the assistant reply. The `Recent turns kept verbatim` setting
(default **2**, range 0–6) protects the newest completed turns, and the user message currently
being answered is always kept. Recent turns are summarized only if there is nothing older left to
fold in and the context still does not fit (the UI then shows a warning).

## 7. How older turns are selected

When total tokens exceed the budget (`MemoryManager.prepare_context`):

1. Take all *completed active* turns except the protected recent ones.
2. Walk from the **oldest** turn forward, adding turns to a batch until the estimated context
   would fall to about **70 %** of the budget (leaving headroom so compression does not fire on
   every message), always taking at least one turn. A batch is also capped at roughly the budget
   size so the summarization request itself stays small.
3. Summarize the batch. If the context is still over budget, repeat (up to 50 rounds).

## 8. How summaries are merged

The provider's `summarize(existing_summary, turns, max_tokens)` receives the **existing summary
and the new turns together** and returns one updated summary. The requested size is
`budget / 5`, clamped to 80–600 tokens.

* **Live LLM:** a fixed instruction prompt asks the model to merge, keep everything still useful,
  remove only duplicates, avoid vague lines, and use headings (Facts, Preferences, Decisions,
  Constraints, Open questions / tasks, Other).
* **Offline demo provider:** a rule-based extractor (regular expressions) tags sentences as
  `[Facts]`, `[Preferences]`, `[Decisions]`, `[Constraints]`, `[Open tasks]`, `[Questions]`,
  `[User said]` or `[Assistant said]`, merges them with the existing bullets, removes duplicates,
  and — only if the size limit forces it — shortens text, then merges the low-priority categories
  (questions and paraphrases) into compact lines, then, as a last resort, drops the *oldest
  fragments* of those low-priority lines. Facts, preferences, decisions, constraints and open
  tasks are never dropped by the offline provider. It is extractive and cruder than an LLM.

If the summary itself grows larger than a quarter of the budget and nothing older remains, one
extra "condense the summary" call is made.

## 9. What the summary should preserve

Concrete facts (names, numbers, dates), user preferences, decisions, questions still open,
constraints, and unfinished tasks. The summary is a separate block: summarized raw messages are
**not** re-sent to the model.

## 10. What happens if summarization fails

If the provider raises an error or returns an empty summary:

* **nothing is committed** — no message is flagged as summarized, the existing summary is unchanged;
* the request is **blocked** (never sent oversized);
* a clear error is shown, and the unsent message is handed back to the user in an expander.

Messages and summary are changed only by `Conversation.apply_summary`, called *after* a
successful summary.

## 11. How budget overflow is prevented

`prepare_context` returns `ok=True` only when the measured total is ≤ budget. Otherwise it
returns `ok=False`, an explanation, and **no messages**, and `app.py` never calls the model.
Cases covered: summarization failure; a single message too large for the budget; a budget smaller
than the system prompt itself; compression that does not converge. Caveat: the check uses the
local *estimate*, so the provider's real count may differ slightly — leave margin below the
model's true limit.

## 12. How token usage is displayed

The dashboard (Streamlit) shows: budget, system prompt, memory summary, recent history, total
input context, remaining tokens, a utilization progress bar, a memory status line, and the
number of compressions and turns summarized. It is rebuilt on every rerun, including after the
user's message, after the assistant reply and after any compression. "Last response" input/output
tokens are shown only when the provider reports them (labelled API-reported); otherwise `n/a`
(always the case in offline mode). Because compression runs *before* the next request, the
dashboard can briefly show over 100 % after a long reply; the status line says so.

## 13. How reset works

The **Reset Conversation** button (with a confirmation dialog) calls `Conversation.reset()` and
clears the UI notices: messages, summary, compression counters and log, last API usage and any
error state. The next message starts from an empty context, so old memory cannot leak in.

## Transcript vs. model context

| | Transcript (UI) | Context (sent to the LLM) |
|---|---|---|
| Contents | **all** messages; summarized ones sit in a collapsed "Earlier messages" panel | system prompt + memory summary + **active** messages only |
| Grows | forever (until reset) | stays within the budget |
| Stored in | `Conversation.messages` | built on demand by `MemoryManager.build_messages` |

## Flow diagram

```
 user sends message
        |
        v
 add to transcript ---> measure context (system + summary + active messages)
                                  |
                      total <= budget? --yes--> send to model --> add reply --> update dashboard
                                  |no
                                  v
              pick oldest completed turns (keep recent turns)
                                  |
                                  v
        provider.summarize(existing summary + picked turns)
                       |                         |
                    success                   failure / empty
                       |                         |
        commit: store merged summary,      change nothing, BLOCK request,
        flag turns as summarized           show error, return message to user
                       |
                       v
          re-measure ---> still over budget? --yes--> repeat (or condense summary,
                                  |no                  or block if nothing left)
                                  v
                          send to model
```

## Known limitations and future improvements

* Token counts are estimates; the real provider count can differ. Using the API-reported
  `prompt_tokens` to calibrate the estimator would help.
* The tokenizer is `cl100k_base`, which only approximates non-OpenAI models.
* The offline summarizer is a rule-based demo; it can miss facts that a regex does not match.
* Summaries are lossy by nature; the full transcript is kept in the UI but is **not** searchable
  by the model after it has been summarized.
* Conversation state lives in Streamlit session state only: refreshing the browser tab or a
  server restart loses it (no database or file persistence).
* Not implemented, possible next steps: persistent storage, retrieval of archived turns,
  streaming responses, per-model context-window presets, multi-level (hierarchical) summaries.
