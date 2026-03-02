"use client";

import { useState } from "react";

export default function CameraCard() {
  const [on, setOn] = useState(false);

  return (
    <div className="rounded-lg border border-mc-border bg-mc-bg-secondary p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-mc-text font-medium flex items-center gap-2">
          <svg className="w-4 h-4 text-mc-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 13v7a2 2 0 01-2 2H7a2 2 0 01-2-2v-7" />
          </svg>
          Camera
        </span>
        <button
          onClick={() => setOn(!on)}
          className={`p-1.5 rounded ${on ? "bg-mc-accent-green/20 text-mc-accent-green" : "hover:bg-mc-bg-tertiary"}`}
          aria-label={on ? "Turn off" : "Turn on"}
        >
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
          </svg>
        </button>
      </div>
      <div className="flex flex-col items-center justify-center py-8">
        {on ? (
          <div className="w-24 h-24 rounded-full bg-mc-bg-tertiary flex items-center justify-center border-2 border-mc-accent">
            <div className="w-16 h-16 rounded-full bg-mc-accent/30 animate-pulse" />
          </div>
        ) : (
          <>
            <svg className="w-16 h-16 text-mc-text-secondary mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
            </svg>
            <span className="text-mc-text font-medium">Camera Off</span>
            <span className="text-mc-text-secondary text-sm text-center mt-1">
              Camera is inactive. Click the power button to start.
            </span>
          </>
        )}
      </div>
    </div>
  );
}
