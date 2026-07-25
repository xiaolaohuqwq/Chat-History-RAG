# 聊天记录 RAG

这是一个本地优先的聊天记录检索应用。解析、SQLite/LanceDB 存储、检索、
上下文组装和终端界面均在本地运行；嵌入、重排和回答生成使用显式配置的云端 API。

## 环境要求

- Python 3.12 和 `uv`
- Node.js 22.19 或更高版本，以及 npm
- 用于转换 QQ 原始导出数据的 `jq`

安装锁定版本的依赖：

```bash
uv sync
npm install
```

## 私有数据处理流程

原始导出文件、规范化 JSONL、数据库、向量、检查点和 `.env` 均已被 Git 忽略。
需要注意云端隐私边界：导入时会将每个窗口的文本发送到 DashScope 生成嵌入；
提问时只会将检索到的证据发送给配置的 OpenAI 兼容 LLM。

显式转换 QQ 导出数据。应用不会自行调用此命令，也不会隐式覆盖输入文件：

```bash
bash qq2jsonl.sh messages
```

然后在不调用 API 的情况下检查成本：

```bash
uv run chat-rag ingest messages.jsonl --dry-run
```

将 `.env.example` 复制为 `.env`，填写服务商密钥，并在确认成本估算后执行正式导入：

```bash
uv run chat-rag smoke-embedding
uv run chat-rag ingest messages.jsonl
```

`smoke-embedding` 只发送三条合成消息，用于验证配置的模型能否返回 1024 维向量。
正式导入支持增量执行，并在每个批次完成后提交。对未变化的输入重复执行不会再次请求嵌入。
仅当嵌入模型、向量维度、规范化版本或窗口版本发生变化时，才需要完整重建向量：

```bash
uv run chat-rag ingest messages.jsonl --rebuild-vectors
```

配置的逻辑批大小为 64。DashScope 适配器会自动拆分为每次最多 20 条输入的请求，
以符合接口的单次请求限制。当账户 TPM 限制要求串行请求时，可设置
`EMBEDDING_CONCURRENCY=1`；被拒绝的多输入批次会自动二分重试，且不会暴露服务商响应正文。

本地生成的数据存放在 `data/`：

```text
data/app.db
data/vectors/
```

## 命令行

```bash
uv run chat-rag search "以前为什么考虑推迟上线？"
uv run chat-rag ask "以前为什么考虑推迟上线，最后形成了什么结论？"
uv run chat-rag inspect MESSAGE_OR_WINDOW_ID
uv run chat-rag stats
uv run chat-rag eval evals.local.jsonl
uv run chat-rag serve --stdio
```

`search` 输出带分数的窗口和原始消息 ID。`ask` 会在需要时执行结构化多查询规划，
按证据用途融合结果，对超大证据使用 map-reduce，并校验最终回答中的所有引用。
`stats` 显示数量、源文件指纹、索引身份、最近一次导入状态、异常行数量和嵌入成本估算，
但不会输出私有消息正文。

可参考 `evals.example.jsonl` 创建私有评测文件。评测文件默认被 Git 忽略，评测结果包含
Recall@20、Recall@50、MRR、日期/发送者覆盖率和端到端检索耗时。

## 终端界面

```bash
npm run tui
```

在编辑器中直接提交内容等同于执行基于证据的 `/ask`。可用命令包括 `/ask`、
`/search`、`/inspect`、`/stats`、`/clear` 和 `/quit`。编辑器支持多行输入和中文输入法
的终端光标定位。按 Escape 取消当前任务，按 Ctrl+C 退出，按 PageUp/PageDown 滚动会话；
不带 ID 的 `/inspect` 会依次查看最近一次回答的引用。

TUI 会把最近 6 条用户和助手消息发送给问答服务，用于改写追问；此前助手给出的结论不会被
当作事实证据。引用 ID 会作为回答元数据保留以供检查，但不会显示在回答正文中。最终回答会
流式写入会话；请求进行期间输入光标会被隐藏，并在完成或取消后恢复。

TUI 会启动一个长期运行的 `uv run chat-rag serve --stdio` 子进程。带版本号、按行分隔的
JSON 协议将 Python 标准输出专用于事件，服务商诊断信息则写入标准错误。

## 服务商配置

嵌入和重排使用 `DASHSCOPE_API_KEY`。回答生成原样使用 OpenAI 兼容的 `API_KEY`、
`BASE_URL` 和 `MODEL`。对于 DeepSeek 官方 API，示例地址为
`https://api.deepseek.com`。中转服务可以配置任意兼容的基础路径；应用不会猜测或自动追加
`/v1`。

请根据实际模型限制设置 `LLM_CONTEXT_WINDOW`。如果输入和输出预算之和超过该限制，
应用会拒绝启动。API 密钥和完整私有提示内容不会被记录，也不会通过 TUI 协议发送。

## 验证

默认测试使用合成数据和模拟传输，不读取 `.env`，也不会使用真实凭据。

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
