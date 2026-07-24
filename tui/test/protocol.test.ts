import { describe, expect, it } from "vitest";

import { parseRpcEvent } from "../src/protocol.js";

describe("RPC protocol", () => {
  it("validates versioned events", () => {
    expect(parseRpcEvent('{"version":1,"id":"a","type":"progress","payload":{"stage":"retrieval"}}').type).toBe("progress");
    expect(() => parseRpcEvent('{"version":2,"id":"a","type":"result","payload":{}}')).toThrow();
    expect(() => parseRpcEvent("not json")).toThrow();
  });
});

