# Chat History RAG

A local-first chat-history retrieval application. Parsing, SQLite/LanceDB storage,
retrieval, context assembly, and the terminal UI run locally. Embedding, reranking,
and answer generation use explicitly configured cloud APIs.

## Prerequisites

- Python 3.12 and `uv`
- Node.js 22.19 or newer and npm
- `jq` for converting a raw QQ export

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
uv run chat-rag ingest messages.jsonl
uv run chat-rag search "以前为什么考虑推迟上线？"
uv run chat-rag ask "以前为什么考虑推迟上线，最后形成了什么结论？"
npm run tui
```

