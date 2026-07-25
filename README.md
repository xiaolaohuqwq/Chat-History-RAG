# 聊天记录 RAG

一个本地优先的聊天记录检索与问答工具。它在本地完成数据解析、存储、混合检索、
证据组装和终端交互，只将嵌入、重排以及回答生成所需的数据发送给显式配置的云端 API。

## 项目简介

聊天记录 RAG 用于从大量 QQ 聊天记录中检索原始消息，并基于可检查的证据回答问题。
它不是单纯的关键词搜索：对于原因、争议、决策过程和后续结果等问题，系统会生成结构化
检索计划，按证据用途融合多个查询的结果，再交给 LLM 综合回答。

### 主要功能

- 将预处理后的聊天记录增量导入 SQLite 和 LanceDB。
- 结合向量检索、SQLite FTS5 和云端重排召回相关消息。
- 支持事实查询、原因分析、观点归纳、时间线和决策过程等问答意图。
- 在 TUI 中保留最近 6 条对话，用于理解“那后来呢？”一类追问。
- 对最终回答中的消息引用进行校验，并支持检查对应原文。
- 对超长证据执行分层压缩，控制 LLM 上下文大小。
- 提供本地检索评测、耗时统计和可审计的导入状态。

## 环境要求

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- Node.js 22.19 或更高版本
- npm
- [`jq`](https://jqlang.github.io/jq/)，用于转换 QQ 原始导出数据
- DashScope API 密钥，用于嵌入和重排
- OpenAI 兼容的 LLM API 密钥

## 安装

1. 安装 Python 和 Node.js 依赖：

   ```bash
   uv sync
   npm install
   ```

2. 创建本地配置文件：

   ```bash
   cp .env.example .env
   ```

3. 在 `.env` 中至少填写以下配置：

   ```dotenv
   DASHSCOPE_API_KEY=你的_DashScope_密钥
   API_KEY=你的_LLM_密钥
   BASE_URL=https://api.deepseek.com
   MODEL=deepseek-v4-flash
   ```

   `BASE_URL` 会原样传给 OpenAI 兼容客户端，程序不会猜测或自动追加 `/v1`。

## 数据准备

原始 QQ 导出目录假设为 `messages/`。转换必须由用户显式执行，应用不会自行覆盖输入：

```bash
bash qq2jsonl.sh messages
```

转换完成后会得到 `messages.jsonl`。先执行试运行，检查有效行数、窗口数量、预计成本和
向量存储空间；此步骤不会调用云端 API：

```bash
uv run chat-rag ingest messages.jsonl --dry-run
```

确认估算后，先用三条合成消息验证嵌入配置，再执行正式导入：

```bash
uv run chat-rag smoke-embedding
uv run chat-rag ingest messages.jsonl
```

正式导入支持断点续传，并在每个批次完成后持久化进度。对未变化的输入重复执行不会再次
生成嵌入。只有在嵌入模型、向量维度、规范化版本或窗口版本变化时才需要完整重建：

```bash
uv run chat-rag ingest messages.jsonl --rebuild-vectors
```

生成的本地状态位于：

```text
data/app.db
data/vectors/
```

## 使用方法

### 终端界面

推荐通过 TUI 使用问答功能：

```bash
npm run tui
```

直接输入问题等同于执行 `/ask`。可用命令如下：

| 命令 | 作用 |
| --- | --- |
| `/ask 问题` | 检索证据并生成回答 |
| `/search 查询` | 直接查看检索结果 |
| `/inspect ID` | 查看指定消息或窗口的原文 |
| `/inspect` | 依次查看最近一次回答引用的消息 |
| `/stats` | 查看索引数量和最近导入状态 |
| `/clear` | 清空当前会话 |
| `/quit` | 退出程序 |

按 Escape 取消当前任务，按 Ctrl+C 退出，按 PageUp/PageDown 滚动会话。TUI 支持中文
输入法和多行输入；回答中的内部引用标号不会显示，但仍作为元数据保留供 `/inspect` 使用。

TUI 会启动一个长期运行的 `uv run chat-rag serve --stdio` 子进程。带版本号、按行分隔的
JSON 协议将 Python 标准输出专用于协议事件，诊断信息写入标准错误。

### 命令行

```bash
# 直接检索并显示带分数的窗口和原始消息
uv run chat-rag search "以前为什么考虑推迟上线？"

# 基于检索证据生成回答
uv run chat-rag ask "以前为什么考虑推迟上线，最后形成了什么结论？"

# 检查原始消息或检索窗口
uv run chat-rag inspect MESSAGE_OR_WINDOW_ID

# 查看索引元数据和统计信息
uv run chat-rag stats
```

### 检索评测

参考 `evals.example.jsonl` 创建私有评测文件：

```bash
uv run chat-rag eval evals.local.jsonl
```

评测文件默认被 Git 忽略。报告包含 Recall@20、Recall@50、MRR、日期/发送者覆盖率和
端到端检索耗时。

## 配置

完整默认值见 `.env.example`。常用配置如下：

| 配置项 | 说明 |
| --- | --- |
| `DASHSCOPE_API_KEY` | 嵌入和重排服务密钥 |
| `EMBEDDING_MODEL` | 嵌入模型名称 |
| `EMBEDDING_DIMENSION` | 嵌入向量维度，默认 1024 |
| `RERANK_MODEL` | 重排模型名称 |
| `API_KEY` | 回答模型的 API 密钥 |
| `BASE_URL` | OpenAI 兼容接口地址 |
| `MODEL` | 回答模型名称 |
| `LLM_CONTEXT_WINDOW` | 模型实际上下文上限 |
| `LLM_MAX_INPUT_TOKENS` | 单次请求的最大输入预算 |
| `LLM_MAX_OUTPUT_TOKENS` | 单次请求的最大输出预算 |
| `RERANK_CANDIDATES` | 进入云端重排的候选窗口数量 |
| `FINAL_EVIDENCE_BLOCKS` | 最终回答最多使用的证据块数量 |

如果输入预算与输出预算之和超过 `LLM_CONTEXT_WINDOW`，程序会拒绝启动。

## 数据与隐私

以下内容已被 Git 忽略：原始导出数据、规范化 JSONL、数据库、向量、检查点、私有评测集
和 `.env`。

- 导入时，每个聊天窗口的文本会发送给 DashScope 生成嵌入。
- 查询时，问题和重排候选会发送给 DashScope。
- 规划追问时，当前问题和最多 6 条会话历史会发送给配置的 OpenAI 兼容 LLM。
- 回答时，只有选中的证据会发送给配置的 OpenAI 兼容 LLM。
- API 密钥、完整私有提示和聊天正文不会通过 TUI 协议输出。
- `stats` 不会打印私有消息正文。

## 支持

当前仓库未配置公开的 Issue Tracker。报告问题时，请通过项目维护渠道提供以下信息：

- 执行的命令和完整错误信息；
- Python、Node.js 和操作系统版本；
- `uv run chat-rag stats` 的非敏感输出；
- 是否使用中转 API，以及对应的 HTTP 状态码。

不要提交 `.env`、API 密钥、聊天原文、数据库或向量文件。

## 路线图

后续可继续改进以下方向：

- 中文关键词拆分和实体提取；
- 聊天事件、观点冲突和决策结果的结构化重建；
- 面向真实问题的私有检索与回答质量评测；
- TUI 中按时间线、观点和证据组织检查结果。

## 贡献

欢迎提交改进。较大的行为变更应先与维护者确认范围，并遵循现有的 Python 后端与
TypeScript TUI 边界。提交前请运行：

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv build
npm run typecheck
npm test
npm run build
```

测试必须使用合成数据或模拟传输，不应读取 `.env`、调用真实凭据或包含私人聊天内容。

## 维护者与致谢

维护者信息以仓库提交记录为准。项目使用 SQLite、LanceDB、Typer、HTTPX 和
`@earendil-works/pi-tui` 等开源组件构建。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 项目状态

项目处于活跃开发阶段，数据格式、RPC 协议和命令行参数仍可能调整。当前版本面向本地私有
聊天记录和单用户工作流，不提供多租户权限控制或托管服务。
