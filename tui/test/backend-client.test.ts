import path from "node:path";
import { fileURLToPath } from "node:url";

import { afterEach, describe, expect, it } from "vitest";

import { BackendClient } from "../src/backend-client.js";

const here = path.dirname(fileURLToPath(import.meta.url));
let client: BackendClient | undefined;

afterEach(async () => {
  await client?.shutdown();
  client = undefined;
});

describe("BackendClient", () => {
  it("keeps one process alive for multiple validated requests", async () => {
    client = new BackendClient(process.execPath, [path.join(here, "fake-backend.mjs")]);
    const eventTypes: string[] = [];
    const answer = await client.request("ask", { question: "中文问题" }, (event) => eventTypes.push(event.type));
    const stats = await client.request("stats", {});

    expect(answer).toMatchObject({ answer: "测试答案 [m1]", citations: ["m1"] });
    expect(stats).toMatchObject({ messages: 3 });
    expect(eventTypes).toContain("progress");
    expect(eventTypes).toContain("answer_delta");
  });

  it("propagates AbortSignal as a cancel request", async () => {
    client = new BackendClient(process.execPath, [path.join(here, "fake-backend.mjs")]);
    const controller = new AbortController();
    const request = client.request("ask", { question: "wait" }, undefined, controller.signal);
    setTimeout(() => controller.abort(), 10);
    await expect(request).rejects.toThrow("cancelled");
    await expect(client.request("stats", {})).resolves.toMatchObject({ messages: 3 });
  });

  it("includes backend diagnostics when the process exits", async () => {
    client = new BackendClient(process.execPath, [path.join(here, "fake-backend.mjs")]);

    await expect(client.request("ask", { question: "crash" })).rejects.toThrow(
      "synthetic backend failure",
    );
  });
});
