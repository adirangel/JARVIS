"use client";

import { useEffect, useState } from "react";
import { getWeather } from "@/lib/api";

type WeatherData = {
  temp_c: number;
  location: string;
  condition: string;
  humidity: number;
  wind_m_s: number;
  feels_like_c: number;
};

export default function WeatherCard() {
  const [data, setData] = useState<WeatherData | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    setLoading(true);
    try {
      const d = await getWeather();
      setData(d);
    } catch {
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 60000);
    return () => clearInterval(id);
  }, []);

  if (loading && !data) {
    return (
      <div className="rounded-lg border border-mc-border bg-mc-bg-secondary p-4">
        <div className="flex items-center justify-between mb-3">
          <span className="text-mc-text font-medium flex items-center gap-2">
            <svg className="w-4 h-4 text-mc-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" />
            </svg>
            Weather
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
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" />
          </svg>
          Weather
        </span>
        <button onClick={fetchData} className="p-1 rounded hover:bg-mc-bg-tertiary" aria-label="Refresh">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>
      </div>
      {data && (
        <>
          <div className="flex items-center justify-between mb-3">
            <div>
              <div className="text-3xl font-semibold text-white">{data.temp_c}°C</div>
              <div className="text-mc-text-secondary text-sm">{data.location}</div>
              <div className="text-mc-text-secondary text-sm">{data.condition}</div>
            </div>
            <svg className="w-16 h-16 text-mc-accent opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" />
            </svg>
          </div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="bg-mc-bg-tertiary rounded px-2 py-1.5 text-center">
              <div className="text-mc-text-secondary">Humidity</div>
              <div className="text-mc-text font-medium">{data.humidity}%</div>
            </div>
            <div className="bg-mc-bg-tertiary rounded px-2 py-1.5 text-center">
              <div className="text-mc-text-secondary">Wind</div>
              <div className="text-mc-text font-medium">{data.wind_m_s} m/s</div>
            </div>
            <div className="bg-mc-bg-tertiary rounded px-2 py-1.5 text-center">
              <div className="text-mc-text-secondary">Feels Like</div>
              <div className="text-mc-text font-medium">{data.feels_like_c}°C</div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
