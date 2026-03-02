"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import Header from "@/components/Header";
import SystemStatsCard from "@/components/Sidebar/SystemStatsCard";
import WeatherCard from "@/components/Sidebar/WeatherCard";
import CameraCard from "@/components/Sidebar/CameraCard";
import UptimeCard from "@/components/Sidebar/UptimeCard";
import AudioVisualizer from "@/components/CenterPanel/AudioVisualizer";
import VoiceControls from "@/components/CenterPanel/VoiceControls";
import ConversationPanel from "@/components/Conversation/ConversationPanel";
import { getConversation, postConversation, getWeather, getHealth, getVoiceStatus } from "@/lib/api";
import { createWebSocket, WsMessage } from "@/lib/websocket";

type Message = { role: string; content: string };

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [backendOnline, setBackendOnline] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState<string>("idle");
  const [weatherSummary, setWeatherSummary] = useState<{ temp_c: number; location: string } | null>(null);
  const destroyWsRef = useRef<(() => void) | null>(null);

  // Health check polling
  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        await getHealth();
        if (mounted) setBackendOnline(true);
      } catch {
        if (mounted) setBackendOnline(false);
      }
    };
    check();
    const id = setInterval(check, 10000);
    return () => { mounted = false; clearInterval(id); };
  }, []);

  // Voice status polling
  useEffect(() => {
    if (!backendOnline) return;
    let mounted = true;
    const check = async () => {
      try {
        const res = await getVoiceStatus();
        if (mounted) setVoiceStatus(res.status);
      } catch {}
    };
    check();
    const id = setInterval(check, 3000);
    return () => { mounted = false; clearInterval(id); };
  }, [backendOnline]);

  // Initial data load
  useEffect(() => {
    if (!backendOnline) return;
    getConversation().then((r) => setMessages(r.messages)).catch(() => {});
    getWeather().then((w) => setWeatherSummary({ temp_c: w.temp_c, location: w.location })).catch(() => {});
  }, [backendOnline]);

  // WebSocket connection
  useEffect(() => {
    if (!backendOnline) return;

    const destroy = createWebSocket(
      (msg: WsMessage) => {
        if (msg.type === "conversation" && msg.messages) {
          setMessages(msg.messages);
        }
        if (msg.type === "status") {
          setVoiceStatus(msg.status);
        }
      },
      (connected) => setWsConnected(connected)
    );
    destroyWsRef.current = destroy;

    return () => {
      destroy();
      destroyWsRef.current = null;
    };
  }, [backendOnline]);

  const handleSend = useCallback(async (text: string) => {
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setLoading(true);
    try {
      const res = await postConversation(text);
      setMessages(res.messages);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "I encountered an error, sir. Please ensure the JARVIS backend is running." },
      ]);
    } finally {
      setLoading(false);
    }
  }, []);

  const statusLabel =
    voiceStatus === "speaking"
      ? "Speaking..."
      : voiceStatus === "listening"
      ? "Listening..."
      : "Awaiting wake word...";

  return (
    <div className="flex flex-col h-screen">
      <Header weatherSummary={weatherSummary} backendOnline={backendOnline} wsConnected={wsConnected} />
      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar */}
        <aside className="w-[280px] flex flex-col gap-4 p-4 overflow-y-auto border-r border-mc-border bg-mc-bg">
          <SystemStatsCard />
          <WeatherCard />
          <CameraCard />
          <UptimeCard />
        </aside>

        {/* Center Panel */}
        <main className="flex-1 flex flex-col items-center justify-center p-8 bg-mc-bg relative">
          {!backendOnline && (
            <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-red-900/80 border border-red-700 text-red-200 px-4 py-2 rounded-lg text-sm flex items-center gap-2 z-10">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
              Backend offline — start the API with: uvicorn api.main:app --port 8000
            </div>
          )}
          <AudioVisualizer status={voiceStatus} active={backendOnline} />
          <h1 className="text-3xl font-bold text-white mt-4 tracking-wider">J.A.R.V.I.S</h1>
          <div className="flex items-center gap-2 mt-2 text-mc-text-secondary text-sm">
            <span
              className={`w-2 h-2 rounded-full ${
                backendOnline
                  ? voiceStatus === "speaking"
                    ? "bg-yellow-400 animate-pulse"
                    : "bg-mc-accent-green animate-pulse"
                  : "bg-red-500"
              }`}
            />
            {backendOnline ? statusLabel : "Offline"}
          </div>
          <VoiceControls voiceStatus={voiceStatus} backendOnline={backendOnline} />
        </main>

        {/* Right Panel */}
        <section className="w-[360px] flex flex-col">
          <ConversationPanel
            messages={messages}
            setMessages={setMessages}
            onSend={handleSend}
            loading={loading}
            backendOnline={backendOnline}
          />
        </section>
      </div>
    </div>
  );
}
