"use client";

import { useEffect, useState } from "react";
import { getUptime } from "@/lib/api";

type UptimeData = {
  uptime_seconds: number;
  uptime_formatted: string;
  session: number;
  commands: number;
  system_load: number;
};

export default function UptimeCard() {
  const [data, setData] = useState<UptimeData | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    setLoading(true);
    try {
      const d = await getUptime();
      setData(d);
    } catch {
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 1000);
    return () => clearInterval(id);
  }, []);

  if (loading && !data) {
    return (
      <div className="rounded-lg border border-mc-border bg-mc-bg-secondary p-4">
        <div className="flex items-center justify-between mb-3">
          <span className="text-mc-text font-medium flex items-center gap-2">
            <svg className="w-4 h-4 text-mc-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            System Uptime
          </span>
        </div>
        <div className="text-mc-text-secondary text-sm">Loading...</div>
      </div>
    );
  }

  const loadLabel = data && data.system_load > 70 ? "High" : data && data.system_load > 40 ? "Moderate" : "Low";

  return (
    <div className="rounded-lg border border-mc-border bg-mc-bg-secondary p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-mc-text font-medium flex items-center gap-2">
          <svg className="w-4 h-4 text-mc-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          System Uptime
        </span>
        <button onClick={fetchData} className="p-1 rounded hover:bg-mc-bg-tertiary" aria-label="Refresh">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>
      </div>
      {data && (
        <>
          <div className="mb-3">
            <div className="text-mc-text-secondary text-sm">System Running For:</div>
            <div className="text-2xl font-semibold text-white">{data.uptime_formatted}</div>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs mb-3">
            <div className="bg-mc-bg-tertiary rounded px-2 py-1.5 text-center">
              <div className="text-mc-text-secondary">Session</div>
              <div className="text-mc-text font-medium">{data.session}</div>
            </div>
            <div className="bg-mc-bg-tertiary rounded px-2 py-1.5 text-center">
              <div className="text-mc-text-secondary">Commands</div>
              <div className="text-mc-text font-medium">{data.commands}</div>
            </div>
          </div>
          <div>
            <div className="text-xs text-mc-text-secondary mb-0.5">System Load</div>
            <div className="h-2 bg-mc-bg-tertiary rounded-full overflow-hidden">
              <div
                className="h-full bg-mc-accent rounded-full"
                style={{ width: `${Math.min(data.system_load, 100)}%` }}
              />
            </div>
            <div className="flex justify-between text-xs mt-0.5">
              <span className="text-mc-text-secondary">{loadLabel}</span>
              <span className="text-mc-text">{data.system_load}</span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
