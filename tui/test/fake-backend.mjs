import readline from "node:readline";

const lines = readline.createInterface({ input: process.stdin });
lines.on("line", (line) => {
  let request;
  try {
    request = JSON.parse(line);
  } catch {
    return;
  }
  const send = (type, payload) =>
    process.stdout.write(JSON.stringify({ version: 1, id: request.id, type, payload }) + "\n");
  if (request.method === "cancel") {
    send("result", { cancelled: true });
  } else if (request.method === "ask") {
    send("progress", { stage: "retrieval" });
    if (request.params.question === "wait") return;
    send("answer_delta", { text: "测试答案 [m1]" });
    send("result", { answer: "测试答案 [m1]", citations: ["m1"] });
  } else if (request.method === "inspect") {
    send("result", { kind: "message", id: request.params.id, sender: "甲", timestamp: "2026", text: "证据" });
  } else {
    send("result", { messages: 3, windows: 2, vectors: 2 });
  }
});
