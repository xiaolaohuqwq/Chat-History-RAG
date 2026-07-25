# Chat History RAG

A local-first chat-history retrieval application. Parsing, SQLite/LanceDB storage,
retrieval, context assembly, and the terminal UI run locally. Embedding, reranking,
and answer generation use explicitly configured cloud APIs.

## Prerequisites

- Python 3.12 and `uv`
- Node.js 22.19 or newer and npm
- `jq` for converting a raw QQ export

Install the locked dependencies:

```bash
uv sync
npm install
```

## Private-data workflow

Raw exports, normalized JSONL, databases, vectors, checkpoints, and `.env` are
ignored by Git. Cloud privacy boundaries are important: ingestion sends every
window's text to DashScope for embedding, while questions send only retrieved
evidence to the configured OpenAI-compatible LLM.

Convert a QQ export explicitly. The application never invokes this command or
overwrites its input implicitly:

```bash
bash qq2jsonl.sh messages
```

Then inspect cost without making API calls:

```bash
uv run chat-rag ingest messages.jsonl --dry-run
```

Copy `.env.example` to `.env`, fill the provider keys, and run real ingestion
only after reviewing the estimate:

```bash
uv run chat-rag smoke-embedding
uv run chat-rag ingest messages.jsonl
```

`smoke-embedding` sends only three synthetic messages and verifies the configured
model returns 1024-dimensional vectors. Real ingestion is incremental and commits
each completed batch. Re-running unchanged input makes no embedding calls. Use a
full vector rebuild when the embedding model, dimension, normalization version,
or windowing version changes:

```bash
uv run chat-rag ingest messages.jsonl --rebuild-vectors
```

The configured logical batch size is 64. The DashScope adapter transparently
splits it into provider requests of at most 20 inputs, which is the endpoint's
enforced per-request limit. Set `EMBEDDING_CONCURRENCY=1` when an account's TPM
limit requires serial provider requests; rejected multi-input batches are
automatically bisected without exposing provider response bodies.

Local generated state is stored under `data/`:

```text
data/app.db
data/vectors/
```

## CLI

```bash
uv run chat-rag search "以前为什么考虑推迟上线？"
uv run chat-rag ask "以前为什么考虑推迟上线，最后形成了什么结论？"
uv run chat-rag inspect MESSAGE_OR_WINDOW_ID
uv run chat-rag stats
uv run chat-rag eval evals.local.jsonl
uv run chat-rag serve --stdio
```

`search` prints scored windows and original message IDs. `ask` performs broad
multi-query retrieval when appropriate, uses map-reduce for oversized evidence,
and validates every final citation. `stats` shows counts, source fingerprint,
index identity, last ingestion status, malformed rows, and estimated embedding
spend without printing private message text.

Create a private evaluation file from `evals.example.jsonl`. Evaluation files
are ignored by default and report Recall@20, Recall@50, MRR, date/sender coverage,
and end-to-end retrieval latency.

## TUI

```bash
npm run tui
```

Plain editor submissions use evidence-based `/ask`. Available commands are
`/ask`, `/search`, `/inspect`, `/stats`, `/clear`, and `/quit`. The editor accepts
multiline input and positions the terminal cursor for Chinese IME use. Escape
cancels active work, Ctrl+C exits, PageUp/PageDown scroll the conversation, and
`/inspect` without an ID cycles through citations from the latest answer.
Citation IDs are retained as answer metadata for inspection but hidden from the
displayed answer text. Final answers stream into the conversation as they are
generated; the input cursor is hidden while a request is active and restored
when it finishes or is cancelled.

The TUI starts one long-lived `uv run chat-rag serve --stdio` child process. The
versioned newline-delimited JSON protocol reserves Python stdout for events and
keeps provider diagnostics on stderr.

## Provider configuration

Embedding and reranking use `DASHSCOPE_API_KEY`. Answer generation uses the
OpenAI-compatible `API_KEY`, `BASE_URL`, and `MODEL` values unchanged. For the
official DeepSeek API, the example uses `https://api.deepseek.com`. A relay can
set any compatible base path; the application does not guess or append `/v1`.

Set `LLM_CONTEXT_WINDOW` to the real model limit. Startup rejects a configuration
where input plus output allowances exceed that limit. No API key or complete
private prompt body is logged or sent over the TUI protocol.

## Verification

Default tests use synthetic data and mock transports; they never use `.env` or
real credentials.

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv build
npm run typecheck
npm test
npm run build
git status --short
git log --oneline --decorate -n 20
```
