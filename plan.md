# Chat History RAG Implementation Plan

## 1. Objective

Build a local RAG application with a TypeScript terminal user interface (TUI) and a Python retrieval backend for semantic retrieval and evidence-based analysis of a large chat-history JSONL file. AI inference must run through cloud APIs; the local machine must only parse data, store records/vectors, search indexes, assemble context, and display results. Keep a CLI surface for ingestion, evaluation, automation, and debugging, but the interactive question-answering experience must be available through the TUI.

The application must support:

- Conversion of the exported `messages.json` into the filtered `messages.jsonl` with the existing `qq2jsonl.sh` script.
- Incremental ingestion of the resulting `messages.jsonl`.
- Semantic retrieval of messages discussing similar topics rather than only exact text matches.
- Hybrid retrieval for names, numbers, URLs, abbreviations, and exact phrases.
- Cloud reranking of retrieved candidates.
- Answers that synthesize conclusions, disagreements, decisions, and changes over time.
- Traceable citations back to original message IDs.
- Context packing and hierarchical summarization that stay safely below a 200K-token LLM context limit.

## 2. Known Input And Constraints

Source and preprocessing pipeline:

- The expected raw QQ export is `messages.json`.
- Run `bash qq2jsonl.sh messages` from the project root. The script requires `jq`, reads `messages.json`, filters empty text plus image/emoji/video placeholders, and writes one normalized record per line to `messages.jsonl`.
- Treat `qq2jsonl.sh` as the required preprocessing step rather than reimplementing its filtering rules inside the Python application.
- The script currently overwrites `messages.jsonl`; never run it implicitly from application startup. The user must invoke conversion explicitly so an existing index input is not replaced unexpectedly.
- At planning time, `qq2jsonl.sh` and the generated `messages.jsonl` exist, while `messages.json` is not present. The implementation must still document the complete raw-export workflow.
- Never commit `messages.json`, `messages.jsonl`, extracted message text, vector data, SQLite data, checkpoints, or evaluation files containing private messages.

Verified generated input:

- Path: `messages.jsonl`
- Size: `40,444,399` bytes
- Rows: `308,768`
- Each line is one JSON object with exactly these observed fields:

```json
{
  "time": "string",
  "uid": "string",
  "name": "string",
  "text": "string"
}
```

Important consequences:

- The file must be processed line by line and must never be loaded fully into memory.
- There is no `conversation_id`; treat the source file as one logical conversation unless future files provide one.
- There is no reply relationship; context expansion can only use source order and time proximity for this dataset.
- `uid` is the stable sender identity; `name` is display metadata and may change.
- Once generated, the RAG application must treat `messages.jsonl` as read-only and never modify it.
- The local computer is low-spec. Do not download or run embedding, reranking, or LLM weights locally.

## 3. Fixed Technical Decisions

Use Python 3.12 as the primary language for ingestion, storage, retrieval, cloud model adapters, context construction, and evaluation. Use `uv` for the Python project, dependency locking, commands, tests, and builds.

Use TypeScript for the interactive TUI. Use Node.js 22.19 or newer because this is the engine requirement of the chosen TUI package. Manage only the TUI dependencies and scripts with npm and commit `package-lock.json`; do not move Python dependency management away from `uv`.

Recommended dependencies:

- `typer`: CLI.
- `pydantic-settings`: validated configuration.
- `dashscope`: `qwen3.7-text-embedding` calls.
- `httpx`: reranker and configurable LLM HTTP clients.
- `openai`: OpenAI-compatible client for the answer/planning LLM.
- `lancedb` and `pyarrow`: persistent local vector index.
- Standard-library `sqlite3`: raw messages, windows, FTS, ingestion state, and metadata.
- `tenacity` or a small explicit retry helper: bounded API retries.
- `pytest`, `pytest-httpx`, `ruff`: tests and quality gates.

TUI dependencies:

- `@earendil-works/pi-tui`: terminal rendering, editor, Markdown, loaders, overlays, key handling, and virtual terminal testing.
- A small schema validator such as `zod` for the Python/TypeScript RPC boundary.
- TypeScript and the minimum build/test tooling needed by the package; keep the TUI dependency set small.

Do not introduce LangChain, LlamaIndex, RAGFlow, ChromaDB, Ollama, or a locally loaded Transformer model. The required pipeline is small enough to implement directly, and direct control over chat windowing and evidence packing is important.

### Git workflow is mandatory

Development must use Git from the first implementation step. The current planning directory is not yet a Git repository, so initialize it before creating project files.

1. Run `git init` and create `.gitignore` before the bootstrap commit.
2. Ignore `.env`, all private chat exports/derived data, `data/`, caches, build output, and other generated artifacts. Commit `.env.example`, `uv.lock`, and `package-lock.json`.
3. Implement one coherent feature or improvement at a time, run its relevant tests and quality checks, inspect the diff, and make a local commit immediately after it passes.
4. Use focused commit messages such as `chore: bootstrap project`, `feat: add streaming ingestion`, `feat: add hybrid retrieval`, `feat: add stdio rpc`, and `feat: add tui chat flow`.
5. Do not defer all work into one final commit and do not mix unrelated refactors with a feature commit. Bug fixes and meaningful later improvements also require their own local commits.
6. No remote repository or push is required. The handoff is complete only when `git log --oneline` shows the feature history and tracked implementation files have no uncommitted changes.

Local storage layout:

```text
data/
  app.db                 # SQLite source-of-truth and FTS index
  vectors/               # LanceDB tables
  checkpoints/           # resumable ingestion/export state if needed
```

## 4. Cloud Models And Configuration

Embedding model:

```text
qwen3.7-text-embedding
```

Verified model-page properties as of 2026-07-25:

- Input price: `CNY 0.5 / 1M tokens`.
- Configurable vector dimension: 256 through 2560.
- Maximum input/context: approximately 131.07K tokens.
- Published limits: 24K RPM and 1M TPM.

Use 1024 dimensions initially. The embedding model, dimension, and text-normalization version form an immutable index identity. If any of them changes, require a full vector rebuild.

Reranking model:

```text
qwen3-rerank
```

Use the documented DashScope endpoint:

```text
POST https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
```

The model page currently lists `CNY 0.5 / 1M tokens` and a 30K context limit. Rerank only the candidate set, never the full corpus.

Required environment variables:

```dotenv
DASHSCOPE_API_KEY=
EMBEDDING_MODEL=qwen3.7-text-embedding
EMBEDDING_DIMENSION=1024
RERANK_MODEL=qwen3-rerank

API_KEY=
BASE_URL=https://api.deepseek.com
MODEL=deepseek-chat
LLM_CONTEXT_WINDOW=200000
LLM_MAX_INPUT_TOKENS=140000
LLM_MAX_OUTPUT_TOKENS=15000
```

The answer/planning LLM adapter must use the OpenAI-compatible chat API. Provider selection and transport configuration use `BASE_URL`, `API_KEY`, and `MODEL`:

- For the official DeepSeek API, set `BASE_URL=https://api.deepseek.com`, supply a DeepSeek API key, and select a supported model such as `deepseek-chat`.
- For a relay, set `BASE_URL` to the relay's OpenAI-compatible base URL, `API_KEY` to its key, and `MODEL` to the relay's model identifier.
- Pass `BASE_URL` through as configured; do not guess whether a relay requires an extra `/v1` suffix.
- Do not add provider-specific DeepSeek branches to retrieval or prompting. Provider details belong only in `llm_client.py`.
- Validate missing configuration at startup of `ask`/TUI, redact keys in errors and logs, and mock the OpenAI-compatible transport in tests.
- Treat 200K as the application's maximum supported context target, not a promise that every configured model supports 200K. Set `LLM_CONTEXT_WINDOW` to the actual limit of `MODEL`; derive input/output budgets from the smaller configured limit and reject internally inconsistent budgets before sending a request.

Optional tuning variables should have validated defaults:

```dotenv
WINDOW_TARGET_TOKENS=500
WINDOW_MAX_TOKENS=800
WINDOW_OVERLAP_MESSAGES=2
SESSION_GAP_MINUTES=20
EMBED_BATCH_SIZE=64
VECTOR_TOP_K_PER_QUERY=40
LEXICAL_TOP_K_PER_QUERY=40
RERANK_CANDIDATES=100
FINAL_EVIDENCE_BLOCKS=30
```

Never log API keys or full request bodies containing private chats. Document the privacy consequence clearly: all embedded window text is sent to DashScope once, while only retrieved evidence is sent to the LLM API during questions.

## 5. Project Structure

Create a normal `src` layout:

```text
pyproject.toml
uv.lock
package.json
package-lock.json
README.md
.env.example
qq2jsonl.sh
src/chat_rag/
  __init__.py
  cli.py
  config.py
  domain.py
  token_estimator.py
  normalize.py
  ingest.py
  windowing.py
  sqlite_store.py
  vector_store.py
  embedding_client.py
  rerank_client.py
  retrieval.py
  context_builder.py
  llm_client.py
  rpc.py
  prompts.py
  service.py
tests/
tui/
  src/
    index.ts
    app.ts
    backend-client.ts
    protocol.ts
    components/
  test/
  tsconfig.json
```

Keep provider-specific API code behind small interfaces so the vector store, retrieval pipeline, and tests do not depend directly on DashScope or OpenAI-compatible response objects.

### Python/TypeScript boundary

The TypeScript TUI must not implement retrieval logic or read SQLite/LanceDB directly. It starts one long-lived Python child process with `uv run chat-rag serve --stdio` and communicates using newline-delimited JSON over stdin/stdout.

- Requests contain a unique ID, method (`ask`, `search`, `inspect`, `stats`, or `cancel`), and validated parameters.
- Responses/events contain the same ID and a type such as `progress`, `retrieval`, `answer_delta`, `result`, or `error`.
- Reserve Python stdout exclusively for protocol messages in stdio-server mode; diagnostics go to stderr.
- Add protocol versioning, validate both sides, tolerate malformed messages without crashing the session, and never include API keys in protocol traffic.
- Propagate cancellation from Escape/Ctrl+C through an `AbortSignal` to the Python request where feasible. Always restore terminal state and terminate the child process on exit.

## 6. SQLite Data Model

At minimum create these tables.

### `messages`

```text
message_id TEXT PRIMARY KEY
source_id TEXT NOT NULL
source_line INTEGER NOT NULL
time_raw TEXT NOT NULL
time_utc TEXT NULL
uid TEXT NOT NULL
name TEXT NOT NULL
text TEXT NOT NULL
content_hash TEXT NOT NULL
UNIQUE(source_id, source_line)
```

Generate `message_id` deterministically from source identity, source line, `uid`, `time`, and text. Exact duplicate records with the same `uid/time/text` may be recorded once in a separate duplicate mapping, but do not collapse merely similar or repeated short messages.

### `windows`

```text
window_id TEXT PRIMARY KEY
source_id TEXT NOT NULL
start_line INTEGER NOT NULL
end_line INTEGER NOT NULL
start_time TEXT NULL
end_time TEXT NULL
text TEXT NOT NULL
estimated_tokens INTEGER NOT NULL
content_hash TEXT NOT NULL
windowing_version TEXT NOT NULL
embedding_model TEXT NULL
embedding_dimension INTEGER NULL
embedded_at TEXT NULL
```

### `window_messages`

```text
window_id TEXT NOT NULL
message_id TEXT NOT NULL
position INTEGER NOT NULL
PRIMARY KEY(window_id, message_id)
```

### `ingestion_runs`

Track source fingerprint, byte size, last completed line, counts, status, model identity, timestamps, and any non-secret error summary. This enables idempotence and recovery.

### Full-text search

Create an FTS5 table over window text or message text. Because default SQLite tokenization is weak for unsegmented Chinese, retrieval must also have a parameterized substring fallback for meaningful Chinese query fragments. Reject empty and punctuation-only lexical queries.

## 7. Normalization And Stable IDs

Normalization for embedding and hashing must be deterministic and versioned:

- Normalize newlines and Unicode whitespace.
- Trim leading/trailing whitespace.
- Preserve wording, punctuation, emoji, URLs, and case.
- Do not rewrite names or message contents.
- Skip empty text after normalization.
- Redact secrets only through an explicit, configurable preprocessing stage; never silently mutate the source file.

Store `NORMALIZATION_VERSION` and `WINDOWING_VERSION`. A version change invalidates affected windows and vectors.

Parse `time` using a small list of explicit known formats discovered from the data during implementation. If parsing fails, preserve `time_raw`, set parsed time to null, keep source order, and record a warning count rather than dropping the message.

## 8. Conversation Windowing

Do not embed every JSONL row independently. Short messages such as “可以” or “就这样” are meaningless without nearby messages.

Windowing algorithm:

1. Stream messages in source order.
2. Start a new session when the parsed gap exceeds 20 minutes. If either timestamp is invalid, rely on source order and token limit.
3. Format each message compactly, retaining identity and time, for example:

```text
[m_ab12 | 2026-07-01 10:15 | 张三(uid123)] 建议推迟上线
```

4. Accumulate approximately 500 tokens, with a hard maximum near 800 estimated tokens.
5. Never split a single short message. Split exceptionally long messages deterministically by paragraphs, then sentence/character boundaries while preserving the parent message ID.
6. Carry the last two messages into the next window as overlap.
7. Do not cross a detected session boundary merely to meet the target size.
8. Merge very small trailing windows into the previous window only when doing so stays below the hard limit and does not cross a session boundary.

The tokenizer for the final LLM is not yet fixed. Implement a conservative local estimator rather than downloading a model tokenizer: count CJK characters conservatively and estimate Latin text by characters per token. Keep generous context headroom. Make the estimator replaceable.

## 9. Embedding Ingestion

Before either ingestion mode, document and verify the explicit preprocessing command:

```bash
bash qq2jsonl.sh messages
```

This command is only necessary when creating or refreshing `messages.jsonl` from `messages.json`. Check that `jq` exists, report conversion failures clearly, and then ingest the JSONL separately. The Python application must not silently run or duplicate this shell conversion.

Implement two passes:

### Dry run

```bash
uv run chat-rag ingest messages.jsonl --dry-run
```

It must report, without calling APIs:

- Valid, malformed, empty, and duplicate row counts.
- Window count and token estimate.
- Estimated embedding cost using `CNY 0.5 / 1M tokens`.
- Estimated vector storage at 1024-dimensional float32.

### Real ingestion

```bash
uv run chat-rag ingest messages.jsonl
```

Requirements:

- Parse and commit in bounded batches.
- Embed only windows whose hash/model/dimension identity is missing or stale.
- Start with embedding batches of 64; make this configurable.
- Before the first bulk run, embed a tiny sample and assert that every returned vector has exactly 1024 dimensions.
- Treat partial API success carefully; map each response to the correct window ID.
- Persist each completed batch so interruption does not lose progress.
- Retry network failures, HTTP 408/409/429, and 5xx with bounded exponential backoff and jitter.
- Fail immediately on authentication/permission errors and validation errors.
- On rerun with unchanged data and configuration, make zero embedding calls.

Do not automatically apply the open-weight Qwen3 `Instruct: ... Query: ...` convention. `qwen3.7-text-embedding` is a different service model. Only distinguish query/document formatting if the current DashScope API documentation explicitly requires it; encode that behavior in one provider adapter and cover it with tests.

## 10. Retrieval Pipeline

Expose two modes:

```bash
uv run chat-rag search "以前为什么考虑推迟上线？"
uv run chat-rag ask "以前为什么考虑推迟上线，最后形成了什么结论？"
```

### Direct search

1. Embed the query.
2. Retrieve vector Top-K from LanceDB.
3. Run FTS5 plus safe substring retrieval for exact entities and Chinese fragments.
4. Merge rankings with Reciprocal Rank Fusion, using a stable constant such as 60.
5. Deduplicate overlapping windows.
6. Optionally call `qwen3-rerank` for the top 100 candidates.
7. Return scored windows with source message IDs, senders, and timestamps.

### Broad analytical questions

For `ask`, use the configured LLM to produce four to eight short retrieval subqueries when the question asks for causes, opinions, disagreements, decisions, outcomes, or changes over time. The planner must return validated JSON and must not invent metadata filters.

Typical subquery coverage should include:

- Supporting views.
- Opposing views.
- Causes and risks.
- Proposed actions.
- Final decisions or outcomes.
- Later reversals or follow-ups.

Retrieve per subquery, union candidates, then apply:

- Exact duplicate and overlapping-window removal.
- Reranking.
- MMR-style redundancy reduction.
- Time-bucket and sender diversity bonuses.
- Recency only when the question asks for current/latest status.

Similarity alone must not allow ten paraphrases from one day to displace a later final decision.

If reranking is unavailable, fall back to fused retrieval scores and report the degraded mode. If retrieval is empty, do not call the answer LLM.

## 11. Context Expansion And Evidence Blocks

For every selected seed window:

1. Resolve its original messages.
2. Add a small number of immediately preceding/following messages within the same detected session.
3. Merge blocks with overlapping source-line ranges.
4. Preserve the full set of message IDs.
5. Keep messages chronological inside each evidence block.

Use a compact format rather than verbose JSON:

```text
<evidence id="e17" start="2026-07-01 10:15" end="2026-07-01 10:23">
[m102 | 张三] 建议推迟上线
[m103 | 李四] 测试还没有完成
</evidence>
```

Every context item passed to an LLM must remain traceable to stored source messages.

## 12. Staying Within A 200K Context Limit

Treat 200K as a hard total budget that may include output. Target at most 140K estimated input tokens.

Default budget:

```text
System instructions and question       5K
Current application-chat summary       5K
Evidence cards                         40K
Critical raw evidence                  70K
Requested output allowance             15K
Safety margin                          65K
```

### Narrow path

If expanded raw evidence is at most about 80K estimated tokens, send it directly in one final answer call.

### Hierarchical path

If raw evidence exceeds 80K:

1. Group evidence by retrieval subquery/topic and then by time.
2. Pack map-stage batches below approximately 30K tokens.
3. Ask the LLM to produce validated `EvidenceCard` objects.
4. Reject cards that cite message IDs not present in their input batch.
5. Feed cards plus the most highly ranked raw evidence into the final reduce call.
6. If still over budget, select cards/evidence using relevance, coverage, novelty, and token cost; never truncate in the middle of a message.

Evidence card schema:

```json
{
  "topic": "是否推迟上线",
  "period": "2026-03 至 2026-05",
  "claims": ["测试未完成", "客户要求按期上线"],
  "proposals": ["推迟一周"],
  "decisions": ["最终推迟一周"],
  "outcomes": [],
  "disagreements": ["销售反对推迟"],
  "source_ids": ["m102", "m118", "m143"],
  "uncertainty": "未找到上线后的复盘"
}
```

Store reusable cards keyed by evidence hashes when useful, but never treat a generated summary as the source of truth. Raw messages remain authoritative.

## 13. Final Answer Contract

The final system prompt must require the LLM to:

- Use only supplied evidence.
- Distinguish a proposal, personal opinion, temporary decision, final decision, and observed outcome.
- Present supporting and opposing views.
- Explain chronological changes when relevant.
- Cite important claims with exact `[message_id]` identifiers.
- State missing evidence and uncertainty explicitly.
- Never infer consensus merely from repetition or silence.
- Avoid claiming that a proposal was implemented without an outcome message.

Recommended answer structure:

```text
结论
主要依据
不同意见
时间变化
不确定或缺失的信息
引用消息
```

After generation, validate that every cited ID exists and was included in the final evidence set. If invalid citations exist, perform at most one bounded repair call; otherwise return a clear citation-validation warning rather than silently accepting fabricated IDs.

## 14. CLI And TUI Surface

Implement at least:

```text
chat-rag ingest PATH [--dry-run] [--rebuild-vectors]
chat-rag search QUERY [--limit N] [--no-rerank]
chat-rag ask QUESTION [--no-rerank]
chat-rag inspect MESSAGE_OR_WINDOW_ID
chat-rag stats
chat-rag eval EVAL_FILE
chat-rag serve --stdio
```

`stats` should show message/window/vector counts, source fingerprint, index identity, last ingestion status, malformed row count, and estimated API spend. It must not expose message text by default.

Run the interactive application through a root npm script:

```bash
npm run tui
```

Implement the TUI in TypeScript with `@earendil-works/pi-tui`, using [the `pi` TUI package](https://github.com/earendil-works/pi/tree/main/packages/tui) as the concrete reference for component composition, differential rendering, keyboard handling, cancellation, and testing. Depend on the published package rather than copying its source. Pin the resolved version in `package-lock.json`; its current package metadata requires Node.js 22.19 or newer.

The TUI must provide:

- A scrollable conversation view rendering answers as Markdown.
- A multiline `Editor` for Chinese questions with correct IME cursor positioning.
- Streaming status for query planning, retrieval, reranking, context reduction, and LLM generation without exposing private prompt bodies.
- A cancellable loader while work is active; Escape cancels the active request and Ctrl+C exits cleanly.
- Citation selection/inspection in an overlay showing message ID, sender, timestamp, and the exact retrieved excerpt.
- Slash commands or equivalent controls for `/search`, `/ask`, `/inspect`, `/stats`, `/clear`, and `/quit`.
- Responsive rendering for narrow terminals. Every custom component must truncate or wrap to the width passed to `render`; no rendered line may exceed it.
- Clear empty, loading, success, cancellation, configuration-error, and provider-error states. Never leave the terminal in raw mode after a crash.

The CLI remains the stable automation and testing interface. The TUI is a client of that same Python service through the stdio protocol, not a second implementation of RAG behavior.

## 15. Testing Strategy

Write tests before implementation behavior for each module. Tests must never use real credentials or make real network requests.

### Unit tests

- JSONL parsing, malformed lines, empty text, Unicode, and very long messages.
- Stable IDs and versioned normalization.
- Time parsing and session-gap boundaries.
- Window target/max size, overlap, and no cross-session merge.
- Conservative token budgeting.
- FTS/substring query validation, including punctuation-only input.
- Reciprocal Rank Fusion and deterministic tie-breaking.
- Overlap merging and context expansion.
- Evidence-card validation and invalid citation rejection.
- Final context always remains below configured input budget.
- OpenAI-compatible LLM configuration for both the DeepSeek base URL and an arbitrary relay URL, including key redaction.
- RPC request/event schema validation, protocol version mismatch, cancellation, and malformed line handling.

### Integration tests

- End-to-end ingest of a synthetic JSONL fixture.
- Re-ingestion is idempotent and performs no second embedding call.
- Interrupted embedding resumes from completed batches.
- Model or dimension change is detected and requires rebuild.
- Mock 429/5xx retries and 401/403 immediate failures.
- Vector + lexical retrieval finds both paraphrased and exact-identifier queries.
- Reranker failure falls back cleanly.
- Empty retrieval skips the LLM.
- Broad evidence triggers map-reduce; narrow evidence uses one answer call.
- Returned citations resolve to original messages.
- A mocked long-lived stdio session supports search, ask progress events, citation inspection, cancellation, and a later request after a recoverable error.

### TUI tests

- Use `VirtualTerminal` from `@earendil-works/pi-tui` for deterministic rendering and input tests.
- Test Chinese/English input, multiline submission, narrow terminal widths, resizing, long answer wrapping, citation overlays, slash commands, cancellation, and clean shutdown.
- Spawn a fake protocol backend; TUI tests must not access private data, Python storage, or cloud APIs.
- Assert that rendered lines never exceed terminal width and that process exit restores terminal state.

### Optional live smoke test

Provide a separate, explicit command that embeds only three synthetic, non-private messages and checks vector dimension. Never run live API smoke tests as part of the default test suite.

## 16. Evaluation

Create an `evals.example.jsonl` format containing:

```json
{
  "query": "为什么考虑推迟上线？",
  "relevant_message_ids": ["m102", "m118"],
  "notes": "应覆盖测试未完成和需求变化"
}
```

The evaluation command should calculate at least:

- Recall@20.
- Recall@50.
- Mean reciprocal rank where applicable.
- Whether results cover multiple dates/senders.
- Retrieval latency excluding API time and including API time separately.

Before tuning dimensions or adding abstractions, manually create 30 to 50 representative queries from real usage. Retrieval quality on this set is the deciding metric.

## 17. Verification And Acceptance Criteria

The implementation is complete only when all of the following hold:

1. `messages.jsonl` is processed as a stream; peak memory does not scale with the entire file and should remain comfortably below 500 MB during parsing/window construction.
2. A dry run reports row/window/token/storage/cost estimates without API calls.
3. The full ingestion can resume after interruption and does not re-embed completed unchanged windows.
4. A repeated unchanged ingestion performs zero embedding calls.
5. Stored vectors all have the configured dimension, initially 1024.
6. Search results include original message IDs, sender names, and timestamps.
7. Semantic queries retrieve paraphrases; exact names/numbers remain discoverable through hybrid retrieval.
8. `ask` handles broad topics with multi-query retrieval and hierarchical summarization.
9. No LLM request exceeds `LLM_MAX_INPUT_TOKENS` according to the conservative estimator.
10. Every answer citation resolves to evidence actually provided to the LLM.
11. Empty retrieval, provider errors, and malformed data fail clearly without leaking secrets or Python tracebacks to normal CLI users.
12. All default tests use mocks and pass without credentials.
13. `messages.json` can be converted explicitly with `bash qq2jsonl.sh messages`, after which the generated JSONL passes schema sampling and is ingested read-only.
14. `API_KEY`, `BASE_URL`, and `MODEL` work unchanged with the official DeepSeek OpenAI-compatible endpoint and with a mocked arbitrary OpenAI-compatible relay.
15. The TypeScript TUI supports asking, progress display, cancellation, Markdown answers, and citation inspection through the Python stdio protocol at both normal and narrow terminal widths.
16. Git history contains focused local commits for the implemented features and improvements; secrets/private data are ignored and tracked implementation files are clean.
17. The final quality gate passes:

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

## 18. Implementation Order

Implement in this order so each stage remains independently verifiable:

1. Initialize Git, add the privacy-safe `.gitignore`, bootstrap the Python `uv` project and test harness, and make the first local commit.
2. Add and document the explicit `messages.json` to `messages.jsonl` preprocessing contract using `qq2jsonl.sh`; test schema validation without committing either data file, then commit.
3. Implement streaming parsing, normalization, SQLite schema, and dry-run statistics; test and commit.
4. Implement deterministic windowing and window/message persistence; test and commit.
5. Implement mocked embedding client, LanceDB storage, dimension validation, and resumable ingestion; test and commit.
6. Implement direct vector search and source inspection; test and commit.
7. Add FTS5 plus Chinese substring fallback and rank fusion; test and commit.
8. Add the mocked/live-separated `qwen3-rerank` adapter; test and commit.
9. Implement overlap merging, adjacent-message expansion, and token-budgeted evidence blocks; test and commit.
10. Add the `BASE_URL`/`API_KEY`/`MODEL` OpenAI-compatible LLM adapter, query planning, and narrow final answers; test both DeepSeek and relay configuration, then commit.
11. Add evidence-card map-reduce for broad/oversized retrieval sets; test and commit.
12. Add citation validation, evaluation CLI, and documentation; test and commit.
13. Implement and test the versioned Python stdio RPC server, including progress events and cancellation; commit.
14. Bootstrap the TypeScript TUI with `@earendil-works/pi-tui`, connect it to the RPC server, and implement the full chat/search/inspect flow; run Python and npm gates, then commit.
15. Run the complete acceptance suite, make focused fix commits for any defects, and finish with clean tracked files and an auditable local Git history.

Do not build a web UI. The required interactive interface is the terminal TUI; ingestion, retrieval evaluation, citations, and context budgeting must remain independently usable through the Python CLI and service layer.
