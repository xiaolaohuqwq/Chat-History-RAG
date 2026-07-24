import type { Terminal } from "@earendil-works/pi-tui";
import { TUI, visibleWidth } from "@earendil-works/pi-tui";
import { describe, expect, it } from "vitest";

import { ChatApp, type Backend } from "../src/app.js";
import type { RpcEvent, RpcMethod } from "../src/protocol.js";

class VirtualTerminal implements Terminal {
  columns = 80;
  rows = 24;
  kittyProtocolActive = false;
  stopped = false;
  output = "";
  start(): void {}
  stop(): void { this.stopped = true; }
  async drainInput(): Promise<void> {}
  write(data: string): void { this.output += data; }
  moveBy(): void {}
  hideCursor(): void {}
  showCursor(): void {}
  clearLine(): void {}
  clearFromCursor(): void {}
  clearScreen(): void {}
  setTitle(): void {}
  setProgress(): void {}
}

class FakeBackend implements Backend {
  calls: Array<{ method: RpcMethod; params: Record<string, unknown> }> = [];
  async request(method: RpcMethod, params: Record<string, unknown>, onEvent?: (event: RpcEvent) => void): Promise<unknown> {
    this.calls.push({ method, params });
    onEvent?.({ version: 1, id: "x", type: "progress", payload: { stage: "retrieval" } });
    if (method === "ask") return { answer: "中文回答 **重点** [m1]", citations: ["m1"] };
    if (method === "inspect") return { kind: "message", id: "m1", sender: "甲", timestamp: "2026", text: "证据原文" };
    if (method === "search") return { results: [] };
    return { messages: 1, windows: 1, vectors: 1 };
  }
  cancel(): void {}
  async shutdown(): Promise<void> {}
}

describe("ChatApp", () => {
  it("accepts Chinese questions and slash commands", async () => {
    const terminal = new VirtualTerminal();
    const tui = new TUI(terminal);
    const backend = new FakeBackend();
    const app = new ChatApp(tui, backend, () => {});

    await app.submit("中文问题");
    await app.submit("/search 编号 ABC-123");
    await app.submit("/stats");
    await app.submit("/inspect m1");

    expect(backend.calls.map((call) => call.method)).toEqual(["ask", "search", "stats", "inspect"]);
  });

  it("never renders a line wider than narrow terminal width", async () => {
    const tui = new TUI(new VirtualTerminal());
    const app = new ChatApp(tui, new FakeBackend(), () => {});
    await app.submit("一个很长的中文问题用来验证窄终端自动换行不会越界");

    for (const width of [20, 32, 80]) {
      expect(app.render(width).every((line) => visibleWidth(line) <= width)).toBe(true);
    }
  });

  it("clears conversation and requests quit cleanly", async () => {
    let quit = false;
    const tui = new TUI(new VirtualTerminal());
    const app = new ChatApp(tui, new FakeBackend(), () => { quit = true; });
    await app.submit("question");
    await app.submit("/clear");
    expect(app.render(40).join("\n")).not.toContain("中文回答");
    await app.submit("/quit");
    expect(quit).toBe(true);
  });
});
