import { z } from "zod";

export const PROTOCOL_VERSION = 1 as const;

export const rpcMethodSchema = z.enum(["ask", "search", "inspect", "stats", "cancel"]);
export type RpcMethod = z.infer<typeof rpcMethodSchema>;

const rpcEventSchema = z.object({
  version: z.literal(PROTOCOL_VERSION),
  id: z.string().nullable(),
  type: z.enum(["progress", "retrieval", "answer_delta", "result", "error"]),
  payload: z.record(z.string(), z.unknown()).optional(),
  code: z.string().optional(),
  message: z.string().optional(),
});

export type RpcEvent = z.infer<typeof rpcEventSchema>;

export function parseRpcEvent(line: string): RpcEvent {
  return rpcEventSchema.parse(JSON.parse(line));
}

export interface RpcRequest {
  version: typeof PROTOCOL_VERSION;
  id: string;
  method: RpcMethod;
  params: Record<string, unknown>;
}

