import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { randomUUID } from "node:crypto";
import readline from "node:readline";

import { parseRpcEvent, PROTOCOL_VERSION, type RpcEvent, type RpcMethod, type RpcRequest } from "./protocol.js";

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  onEvent?: (event: RpcEvent) => void;
  removeAbortListener?: () => void;
}

export class BackendClient {
  private readonly child: ChildProcessWithoutNullStreams;
  private readonly pending = new Map<string, PendingRequest>();
  private diagnostics = "";
  private closed = false;

  constructor(command = "uv", args = ["run", "chat-rag", "serve", "--stdio"]) {
    this.child = spawn(command, args, {
      cwd: process.cwd(),
      env: process.env,
      stdio: ["pipe", "pipe", "pipe"],
    });
    const lines = readline.createInterface({ input: this.child.stdout });
    lines.on("line", (line) => this.handleLine(line));
    this.child.stderr.setEncoding("utf8");
    this.child.stderr.on("data", (chunk: string) => {
      this.diagnostics = (this.diagnostics + chunk).slice(-4000);
    });
    this.child.on("exit", () => {
      this.closed = true;
      for (const pending of this.pending.values()) {
        pending.reject(new Error("Python backend exited"));
      }
      this.pending.clear();
    });
    this.child.on("error", (error) => {
      for (const pending of this.pending.values()) pending.reject(error);
      this.pending.clear();
    });
  }

  request(
    method: RpcMethod,
    params: Record<string, unknown>,
    onEvent?: (event: RpcEvent) => void,
    signal?: AbortSignal,
  ): Promise<unknown> {
    if (this.closed) return Promise.reject(new Error("Python backend is not running"));
    if (signal?.aborted) return Promise.reject(new Error("request cancelled"));
    const id = randomUUID();
    const request: RpcRequest = { version: PROTOCOL_VERSION, id, method, params };
    return new Promise((resolve, reject) => {
      const pending: PendingRequest = { resolve, reject };
      if (onEvent !== undefined) pending.onEvent = onEvent;
      if (signal !== undefined) {
        const abort = () => {
          this.sendCancel(id);
          reject(new Error("request cancelled"));
          this.pending.delete(id);
        };
        signal.addEventListener("abort", abort, { once: true });
        pending.removeAbortListener = () => signal.removeEventListener("abort", abort);
      }
      this.pending.set(id, pending);
      this.child.stdin.write(JSON.stringify(request) + "\n");
    });
  }

  cancel(): void {
    for (const id of this.pending.keys()) this.sendCancel(id);
  }

  async shutdown(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    for (const pending of this.pending.values()) pending.reject(new Error("backend shutdown"));
    this.pending.clear();
    this.child.stdin.end();
    this.child.kill("SIGTERM");
    await Promise.race([
      new Promise<void>((resolve) => this.child.once("exit", () => resolve())),
      new Promise<void>((resolve) => setTimeout(resolve, 1000)),
    ]);
    if (this.child.exitCode === null) this.child.kill("SIGKILL");
  }

  private sendCancel(requestId: string): void {
    if (this.closed) return;
    const cancel: RpcRequest = {
      version: PROTOCOL_VERSION,
      id: randomUUID(),
      method: "cancel",
      params: { request_id: requestId },
    };
    this.child.stdin.write(JSON.stringify(cancel) + "\n");
  }

  private handleLine(line: string): void {
    let event: RpcEvent;
    try {
      event = parseRpcEvent(line);
    } catch {
      return;
    }
    if (event.id === null) return;
    const pending = this.pending.get(event.id);
    if (pending === undefined) return;
    pending.onEvent?.(event);
    if (event.type === "result") {
      pending.removeAbortListener?.();
      this.pending.delete(event.id);
      pending.resolve(event.payload ?? {});
    } else if (event.type === "error") {
      pending.removeAbortListener?.();
      this.pending.delete(event.id);
      pending.reject(new Error(event.message ?? event.code ?? "backend request failed"));
    }
  }
}

