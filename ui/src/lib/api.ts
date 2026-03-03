const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function fetchApi<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options?.headers },
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function getHealth() {
  return fetchApi<{ status: string }>("/api/health");
}

export async function getSystem() {
  return fetchApi<{
    cpu_percent: number;
    ram: { used_gb: number; total_gb: number; percent: number };
    disk: { used_gb: number; total_gb: number; percent: number };
    os: string;
  }>("/api/system");
}

export async function getWeather() {
  return fetchApi<{
    temp_c: number;
    location: string;
    condition: string;
    humidity: number;
    wind_m_s: number;
    feels_like_c: number;
  }>("/api/weather");
}

export async function getUptime() {
  return fetchApi<{
    uptime_seconds: number;
    uptime_formatted: string;
    session: number;
    commands: number;
    system_load: number;
  }>("/api/uptime");
}

export async function getVoiceStatus() {
  return fetchApi<{ status: string; message: string }>("/api/voice/status");
}

export async function getConversation() {
  return fetchApi<{ messages: { role: string; content: string }[] }>("/api/conversation");
}

export async function postConversation(text: string) {
  return fetchApi<{ response: string; messages: { role: string; content: string }[] }>(
    "/api/conversation",
    { method: "POST", body: JSON.stringify({ text }) }
  );
}

/**
 * Stream a conversation response via SSE. Calls onToken for each chunk,
 * onDone when complete. Returns an abort function.
 */
export function streamConversation(
  text: string,
  onToken: (token: string) => void,
  onDone: (fullResponse: string) => void,
  onError?: (err: Error) => void
): () => void {
  const controller = new AbortController();
  const url = `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/conversation/stream`;

  (async () => {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
        signal: controller.signal,
      });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response body");
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.done) {
              onDone(data.response || "");
            } else if (data.token) {
              onToken(data.token);
            }
          } catch {}
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name !== "AbortError") {
        onError?.(err);
      }
    }
  })();

  return () => controller.abort();
}

export async function clearConversation() {
  return fetchApi<{ ok: boolean }>("/api/conversation", { method: "DELETE" });
}

export function getWsUrl(): string {
  const base = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  return base.replace(/^http/, "ws") + "/ws";
}
