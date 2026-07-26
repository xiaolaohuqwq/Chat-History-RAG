# Chat History RAG

Chat History RAG 用于从大量消息记录（比如 QQ）检索相关信息，返回自然语言问题的查询结果。
对导入消息文件进行数据提取、文本分割和向量化，构建成一个可检索的数据库，用户输入问题后进行多次结构化检索，最后整理结果调用 LLM 综合回答。

## Environment

- Python 3.12
- uv
- Node.js 22.19
- npm

若使用 QQ 导出数据：
- jq
- [qq-chat-exporter](https://github.com/shuakami/qq-chat-exporter)

## Installation

1. 安装 Python 和 Node.js 依赖：

   ```bash
   uv sync
   npm install
   ```

2. 配置文件：

   ```bash
   cp .env.example .env
   ```

`.env` 至少填写以下配置：

   ```dotenv
   DASHSCOPE_API_KEY
   API_KEY
   BASE_URL
   MODEL
   ```

已知支持 qwen3.7-text-embedding, qwen3-rerank, deepseek-v4-flash 或符合 OpenAI 兼容格式的第三方 API 。

## Initialization

使用 [qq-chat-exporter](https://github.com/shuakami/qq-chat-exporter) 导出 qq 聊天记录文件 `messages.json`。执行：

```bash
bash qq2jsonl.sh messages
```

得到 `messages.jsonl`。
如果是其他方式得到的 `jsonl` 文件，需要满足格式：
 `{"time":"2026-01-01 00:00:00","uid":"1","name":"名字","text":"内容"}`

可以先验证配置试运行：

```bash
uv run chat-rag smoke-embedding
```

再执行正式导入：

```bash
uv run chat-rag ingest messages.jsonl
```

生成的本地状态位于：

```text
data/app.db
data/vectors/
```

插入新数据不会覆盖旧数据，要删除旧数据则执行：

```bash
uv run chat-rag reset
```

## Usage

### TUI

```bash
npm run tui
```

直接输入文本默认执行 `/ask`。可用命令如下：

| 命令 | 作用 |
| --- | --- |
| `/ask 问题` | 检索证据并生成回答 |
| `/search 查询` | 查看检索结果 |
| `/inspect ID` | 查看指定消息或窗口的原文 |
| `/inspect` | 依次查看最近一次回答引用的消息 |
| `/stats` | 查看索引数量和最近导入状态 |
| `/clear` | 清空当前会话 |
| `/quit` | 退出程序 |

### CLI

```bash
# 基于检索信息生成回答
uv run chat-rag ask "问题"

# 检索原始消息
uv run chat-rag search "问题"

# 检查原始消息或检索窗口
uv run chat-rag inspect MESSAGE_OR_WINDOW_ID

# 查看索引元数据和统计信息
uv run chat-rag stats

# 删除本地信息和向量索引
uv run chat-rag reset
```

## Config

`.env` 中配置如下：

| 配置项 | 说明 |
| --- | --- |
| `DASHSCOPE_API_KEY` | 嵌入和重排服务密钥 |
| `EMBEDDING_MODEL` | 嵌入模型名称 |
| `EMBEDDING_DIMENSION` | 嵌入向量维度 |
| `RERANK_MODEL` | 重排模型名称 |
| `API_KEY` | 回答模型密钥 |
| `BASE_URL` | OpenAI 兼容接口地址 |
| `MODEL` | 回答模型名称 |
| `LLM_CONTEXT_WINDOW` | 模型实际上下文上限 |
| `LLM_MAX_INPUT_TOKENS` | 单次请求的最大输入预算 |
| `LLM_MAX_OUTPUT_TOKENS` | 单次请求的最大输出预算 |
| `RERANK_CANDIDATES` | 重排的候选窗口数量 |
| `FINAL_EVIDENCE_BLOCKS` | 最多使用的证据块数量 |

## Acknowledgment

本项目使用 SQLite、LanceDB、Typer、HTTPX 和
`@earendil-works/pi-tui` 等开源组件构建。

本项目使用 gpt-5.6 辅助开发，尽可能 Vibecode cleanup，酌情评价。

QQ 聊天记录导出依赖开源项目 [qq-chat-exporter](https://github.com/shuakami/qq-chat-exporter)。


## License

[MIT](https://choosealicense.com/licenses/mit/)
