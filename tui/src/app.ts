import {
  CancellableLoader,
  CombinedAutocompleteProvider,
  Editor,
  Key,
  Markdown,
  matchesKey,
  Text,
  truncateToWidth,
  wrapTextWithAnsi,
  type Component,
  type Focusable,
  type TUI,
} from "@earendil-works/pi-tui";

import type { RpcEvent, RpcMethod } from "./protocol.js";
import { colors, editorTheme, markdownTheme } from "./theme.js";

export interface Backend {
  request(
    method: RpcMethod,
    params: Record<string, unknown>,
    onEvent?: (event: RpcEvent) => void,
    signal?: AbortSignal,
  ): Promise<unknown>;
  cancel(): void;
  shutdown(): Promise<void>;
}

interface ConversationItem {
  role: "user" | "assistant" | "system";
  text: string;
}

const stageLabels: Record<string, string> = {
  planning: "规划检索问题",
  retrieval: "检索相关消息",
  reranking: "重排候选证据",
  context_reduction: "压缩长证据",
  generation: "生成证据回答",
};

const citationLabelPattern = /\[[A-Za-z][A-Za-z0-9_-]*(?:\s*[,|]\s*[A-Za-z][A-Za-z0-9_-]*)*\]/g;
const partialCitationLabelPattern = /\[[A-Za-z][A-Za-z0-9_,|\s-]*$/;

function hideCitationLabels(text: string): string {
  return text
    .replace(citationLabelPattern, "")
    .replace(partialCitationLabelPattern, "")
    .replace(/\s+([，。！？；：,.!?;:])/g, "$1");
}

class CitationOverlay implements Component {
  constructor(private readonly text: string, private readonly close: () => void) {}
  render(width: number): string[] {
    const inner = Math.max(1, width - 4);
    return [
      truncateToWidth(colors.bold("引用详情"), width),
      ...wrapTextWithAnsi(this.text, inner).map((line) => truncateToWidth(`  ${line}`, width)),
      truncateToWidth(colors.dim("Esc 关闭"), width),
    ];
  }
  handleInput(data: string): void {
    if (matchesKey(data, Key.escape)) this.close();
  }
  invalidate(): void {}
}

export class ChatApp implements Component, Focusable {
  private readonly editor: Editor;
  private readonly items: ConversationItem[] = [];
  private loader: CancellableLoader | undefined;
  private status = "就绪";
  private streamText = "";
  private scrollOffset = 0;
  private citations: string[] = [];
  private citationIndex = 0;
  private _focused = false;

  constructor(
    private readonly tui: TUI,
    private readonly backend: Backend,
    private readonly onQuit: () => void,
  ) {
    this.editor = new Editor(tui, editorTheme, { paddingX: 1, autocompleteMaxVisible: 6 });
    this.editor.setAutocompleteProvider(
      new CombinedAutocompleteProvider(
        [
          { name: "ask", description: "证据问答", argumentHint: "问题" },
          { name: "search", description: "直接检索", argumentHint: "查询" },
          { name: "inspect", description: "查看引用", argumentHint: "消息或窗口 ID" },
          { name: "stats", description: "索引状态" },
          { name: "clear", description: "清空会话" },
          { name: "quit", description: "退出" },
        ],
        process.cwd(),
      ),
    );
    this.editor.onSubmit = (text) => void this.submit(text);
  }

  get focused(): boolean { return this._focused; }
  set focused(value: boolean) {
    this._focused = value;
    this.editor.focused = value;
  }

  async submit(raw: string): Promise<void> {
    const input = raw.trim();
    if (!input || this.loader !== undefined) return;
    this.editor.addToHistory(input);
    this.editor.setText("");
    if (input === "/clear") {
      this.items.length = 0;
      this.citations = [];
      this.status = "已清空";
      this.tui.requestRender();
      return;
    }
    if (input === "/quit") {
      this.onQuit();
      return;
    }
    const [command, ...rest] = input.startsWith("/") ? input.slice(1).split(/\s+/) : ["ask", input];
    const argument = rest.join(" ").trim();
    if (command === "ask" || command === "search") {
      if (!argument) return this.addError(`/${command} 需要内容`);
      this.items.push({ role: "user", text: argument });
      await this.run(command, command === "ask" ? { question: argument } : { query: argument });
    } else if (command === "stats") {
      await this.run("stats", {});
    } else if (command === "inspect") {
      const id = argument || this.nextCitation();
      if (!id) return this.addError("当前没有可检查的引用");
      await this.inspect(id);
    } else {
      this.addError(`未知命令: /${command}`);
    }
  }

  render(width: number): string[] {
    const safeWidth = Math.max(1, width);
    const editorLines = this.editor.render(safeWidth);
    const header = truncateToWidth(`${colors.bold("Chat History RAG")}  ${colors.dim(this.status)}`, safeWidth);
    const conversation: string[] = [];
    for (const item of this.items) {
      if (item.role === "assistant") {
        conversation.push(...new Markdown(item.text, 1, 0, markdownTheme).render(safeWidth));
      } else {
        const prefix = item.role === "user" ? colors.cyan("查询: ") : colors.red("提示: ");
        conversation.push(...wrapTextWithAnsi(prefix + item.text, safeWidth));
      }
      conversation.push("");
    }
    if (this.streamText) {
      conversation.push(
        ...new Markdown(hideCitationLabels(this.streamText), 1, 0, markdownTheme).render(safeWidth),
      );
    }
    const loaderLines = this.loader?.render(safeWidth) ?? [];
    const viewport = Math.max(3, this.tui.terminal.rows - editorLines.length - loaderLines.length - 2);
    const end = Math.max(0, conversation.length - this.scrollOffset);
    const start = Math.max(0, end - viewport);
    return [header, ...conversation.slice(start, end), ...loaderLines, ...editorLines].map((line) => truncateToWidth(line, safeWidth));
  }

  handleInput(data: string): void {
    if (matchesKey(data, Key.ctrl("c"))) {
      this.onQuit();
    } else if (this.loader !== undefined && matchesKey(data, Key.escape)) {
      this.loader.handleInput(data);
    } else if (matchesKey(data, Key.pageUp)) {
      this.scrollOffset += Math.max(1, Math.floor(this.tui.terminal.rows / 2));
      this.tui.requestRender();
    } else if (matchesKey(data, Key.pageDown)) {
      this.scrollOffset = Math.max(0, this.scrollOffset - Math.max(1, Math.floor(this.tui.terminal.rows / 2)));
      this.tui.requestRender();
    } else {
      this.editor.handleInput(data);
    }
  }

  invalidate(): void {
    this.editor.invalidate();
  }

  private async run(method: "ask" | "search" | "stats", params: Record<string, unknown>): Promise<void> {
    const showHardwareCursor = this.tui.getShowHardwareCursor();
    this.tui.setShowHardwareCursor(false);
    this.loader = new CancellableLoader(this.tui, colors.cyan, colors.dim, "处理中");
    this.loader.start();
    this.editor.disableSubmit = true;
    this.status = "处理中";
    this.streamText = "";
    this.scrollOffset = 0;
    try {
      const result = await this.backend.request(
        method,
        params,
        (event) => this.handleEvent(event),
        this.loader.signal,
      ) as Record<string, unknown>;
      if (method === "ask") {
        const answer = typeof result.answer === "string" ? result.answer : this.streamText;
        this.items.push({ role: "assistant", text: answer || "未返回回答" });
        this.citations = Array.isArray(result.citations) ? result.citations.filter((id): id is string => typeof id === "string") : [];
      } else if (method === "search") {
        this.items.push({ role: "assistant", text: this.formatSearchResults(result.results) });
      } else {
        this.items.push({ role: "assistant", text: this.formatStats(result) });
      }
      this.status = "完成";
    } catch (error) {
      const message = error instanceof Error ? error.message : "未知错误";
      this.items.push({ role: "system", text: message.includes("cancel") ? "请求已取消" : message });
      this.status = message.includes("cancel") ? "已取消" : "错误";
    } finally {
      this.streamText = "";
      this.loader.stop();
      this.loader.dispose();
      this.loader = undefined;
      this.editor.disableSubmit = false;
      this.tui.setShowHardwareCursor(showHardwareCursor);
      this.tui.requestRender();
    }
  }

  private handleEvent(event: RpcEvent): void {
    if (event.type === "progress") {
      const stage = event.payload?.stage;
      this.status = typeof stage === "string" ? (stageLabels[stage] ?? stage) : "处理中";
      this.loader?.setMessage(this.status);
    } else if (event.type === "answer_delta") {
      const text = event.payload?.text;
      if (typeof text === "string") this.streamText += text;
    }
    this.tui.requestRender();
  }

  private async inspect(id: string): Promise<void> {
    try {
      const result = await this.backend.request("inspect", { id }) as Record<string, unknown>;
      const detail = [result.id, result.sender, result.timestamp, result.text].filter((value) => typeof value === "string").join("\n");
      let handle: ReturnType<TUI["showOverlay"]>;
      const overlay = new CitationOverlay(detail, () => handle.hide());
      handle = this.tui.showOverlay(overlay, { width: "80%", maxHeight: "70%", margin: 1 });
      this.tui.requestRender();
    } catch (error) {
      this.addError(error instanceof Error ? error.message : "引用检查失败");
    }
  }

  private nextCitation(): string {
    if (this.citations.length === 0) return "";
    const id = this.citations[this.citationIndex % this.citations.length] ?? "";
    this.citationIndex += 1;
    return id;
  }

  private formatSearchResults(value: unknown): string {
    if (!Array.isArray(value) || value.length === 0) return "未检索到结果。";
    return value.map((item, index) => {
      const row = item as Record<string, unknown>;
      const messages = Array.isArray(row.messages) ? row.messages as Array<Record<string, unknown>> : [];
      const excerpts = messages.map((message) => `- [${String(message.id)}] ${String(message.sender)}: ${String(message.text)}`).join("\n");
      return `### ${index + 1}. ${String(row.window_id)}\n${excerpts}`;
    }).join("\n\n");
  }

  private formatStats(result: Record<string, unknown>): string {
    return `消息: ${String(result.messages ?? 0)}  窗口: ${String(result.windows ?? 0)}  向量: ${String(result.vectors ?? 0)}`;
  }

  private addError(message: string): void {
    this.items.push({ role: "system", text: message });
    this.status = "错误";
    this.tui.requestRender();
  }
}
