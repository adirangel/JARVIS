"use client";

import { useEffect, useState } from "react";
import { getSystem } from "@/lib/api";

type SystemData = {
  cpu_percent: number;
  ram: { used_gb: number; total_gb: number; percent: number };
  disk: { used_gb: number; total_gb: number; percent: number };
  os: string;
};

export default function SystemStatsCard() {
  const [data, setData] = useState<SystemData | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    setLoading(true);
    try {
      const d = await getSystem();
      setData(d);
    } catch {
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 5000);
    return () => clearInterval(id);
  }, []);

  if (loading && !data) {
    return (
      <div className="rounded-lg border border-mc-border bg-mc-bg-secondary p-4">
        <div className="flex items-center justify-between mb-3">
          <span className="text-mc-text font-medium flex items-center gap-2">
            <svg className="w-4 h-4 text-mc-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            System Stats
          </span>
        </div>
        <div className="text-mc-text-secondary text-sm">Loading...</div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-mc-border bg-mc-bg-secondary p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-mc-text font-medium flex items-center gap-2">
          <svg className="w-4 h-4 text-mc-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          System Stats
        </span>
        <button onClick={fetchData} className="p-1 rounded hover:bg-mc-bg-tertiary" aria-label="Refresh">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>
      </div>
      {data && (
        <>
          <div className="space-y-2 mb-3">
            <div>
              <div className="text-xs text-mc-text-secondary mb-0.5">CPU Usage</div>
              <div className="h-2 bg-mc-bg-tertiary rounded-full overflow-hidden">
                <div className="h-full bg-mc-accent rounded-full" style={{ width: `${data.cpu_percent}%` }} />
              </div>
            </div>
            <div>
              <div className="text-xs text-mc-text-secondary mb-0.5">RAM Usage</div>
              <div className="h-2 bg-mc-bg-tertiary rounded-full overflow-hidden">
                <div className="h-full bg-mc-accent rounded-full" style={{ width: `${data.ram.percent}%` }} />
              </div>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="bg-mc-bg-tertiary rounded px-2 py-1.5 text-center">
              <div className="text-mc-text-secondary">CPU</div>
              <div className="text-mc-text font-medium">{data.cpu_percent}%</div>
            </div>
            <div className="bg-mc-bg-tertiary rounded px-2 py-1.5 text-center">
              <div className="text-mc-text-secondary">Memory</div>
              <div className="text-mc-text font-medium">{data.ram.percent}%</div>
            </div>
            <div className="bg-mc-bg-tertiary rounded px-2 py-1.5 text-center">
              <div className="text-mc-text-secondary">Disk</div>
              <div className="text-mc-text font-medium">{data.disk.used_gb}/{data.disk.total_gb} GB</div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
