import { matchesKey, ProcessTerminal, TUI } from "@earendil-works/pi-tui";

import { ChatApp } from "./app.js";
import { BackendClient } from "./backend-client.js";

const terminal = new ProcessTerminal();
const tui = new TUI(terminal, true);
const backend = new BackendClient();
let shuttingDown = false;

async function shutdown(code = 0): Promise<void> {
  if (shuttingDown) return;
  shuttingDown = true;
  tui.stop();
  await backend.shutdown();
  await terminal.drainInput(250, 30);
  process.exitCode = code;
}

const app = new ChatApp(tui, backend, () => void shutdown());
tui.addChild(app);
tui.setFocus(app);
tui.addInputListener((data) => {
  if (matchesKey(data, "ctrl+c")) {
    void shutdown();
    return { consume: true };
  }
  return undefined;
});

process.once("SIGTERM", () => void shutdown());
process.once("uncaughtException", () => void shutdown(1));
process.once("unhandledRejection", () => void shutdown(1));

terminal.setTitle("Chat History RAG");
tui.start();
