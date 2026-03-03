import { getWsUrl } from "./api";

export type WsMessage =
  | { type: "status"; status: string }
  | { type: "conversation"; messages: { role: string; content: string }[] }
  | { type: "stream"; token: string; done: boolean }
  | { type: "system"; cpu: number; ram: number }
  | { type: "pong" };

/**
 * Creates a WebSocket with automatic reconnection.
 * Returns a cleanup function to close the socket and stop reconnecting.
 */
export function createWebSocket(
  onMessage: (msg: WsMessage) => void,
  onConnChange?: (connected: boolean) => void
): () => void {
  let ws: WebSocket | null = null;
  let pingInterval: ReturnType<typeof setInterval> | null = null;
  let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
  let destroyed = false;

  function connect() {
    if (destroyed) return;
    try {
      ws = new WebSocket(getWsUrl());
    } catch {
      scheduleReconnect();
      return;
    }

    ws.onopen = () => {
      onConnChange?.(true);
      pingInterval = setInterval(() => {
        if (ws?.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "ping" }));
        }
      }, 30000);
    };

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        onMessage(msg);
      } catch {}
    };

    ws.onclose = () => {
      cleanup();
      onConnChange?.(false);
      scheduleReconnect();
    };

    ws.onerror = () => {
      ws?.close();
    };
  }

  function cleanup() {
    if (pingInterval) {
      clearInterval(pingInterval);
      pingInterval = null;
    }
  }

  function scheduleReconnect() {
    if (destroyed) return;
    reconnectTimeout = setTimeout(connect, 3000);
  }

  connect();

  // Return a destroy function
  return () => {
    destroyed = true;
    cleanup();
    if (reconnectTimeout) clearTimeout(reconnectTimeout);
    if (ws) {
      ws.onclose = null;
      ws.close();
    }
  };
}
