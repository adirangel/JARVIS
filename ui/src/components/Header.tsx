"use client";

import { useEffect, useState } from "react";

export default function Header({
  weatherSummary,
  backendOnline,
  wsConnected,
}: {
  weatherSummary?: { temp_c: number; location: string } | null;
  backendOnline?: boolean;
  wsConnected?: boolean;
}) {
  const [time, setTime] = useState("");
  const [date, setDate] = useState("");

  useEffect(() => {
    const update = () => {
      const now = new Date();
      setTime(now.toLocaleTimeString("en-US", { hour12: true, hour: "numeric", minute: "2-digit", second: "2-digit" }));
      setDate(now.toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" }));
    };
    update();
    const id = setInterval(update, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <header className="flex items-center justify-between px-6 py-3 bg-mc-bg-secondary border-b border-mc-border">
      <div className="flex items-center gap-3">
        <span className="text-white font-semibold text-lg tracking-wide">J.A.R.V.I.S</span>
        <span className="flex items-center gap-1.5 text-sm">
          <span
            className={`w-2 h-2 rounded-full ${
              backendOnline ? "bg-mc-accent-green animate-pulse" : "bg-red-500"
            }`}
          />
          <span className={backendOnline ? "text-mc-accent-green" : "text-red-400"}>
            {backendOnline ? "Online" : "Offline"}
          </span>
        </span>
        {backendOnline && (
          <span className="flex items-center gap-1 text-xs text-mc-text-secondary">
            <span
              className={`w-1.5 h-1.5 rounded-full ${wsConnected ? "bg-mc-accent" : "bg-yellow-500"}`}
            />
            {wsConnected ? "WS" : "Polling"}
          </span>
        )}
      </div>
      <div className="flex items-center gap-6 text-mc-text-secondary text-sm">
        <span className="flex items-center gap-1.5">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          {time} | {date}
        </span>
        {weatherSummary && (
          <span className="flex items-center gap-1.5">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" />
            </svg>
            {weatherSummary.temp_c}°C {weatherSummary.location}
          </span>
        )}
        <button className="p-1 rounded hover:bg-mc-bg-tertiary" aria-label="Settings">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </button>
      </div>
    </header>
  );
}
