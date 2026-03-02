"use client";

type Props = {
  voiceStatus?: string;
  backendOnline?: boolean;
};

export default function VoiceControls({ voiceStatus = "idle", backendOnline = false }: Props) {
  const isMicActive = voiceStatus === "listening" || voiceStatus === "speaking";

  return (
    <div className="flex items-center justify-center gap-4 mt-6">
      <button
        className="p-3 rounded-full bg-mc-bg-tertiary hover:bg-mc-border transition-colors disabled:opacity-40"
        aria-label="Camera"
        disabled={!backendOnline}
      >
        <svg className="w-5 h-5 text-mc-text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
        </svg>
      </button>
      <button
        className={`p-4 rounded-full transition-all ${
          isMicActive && backendOnline
            ? "bg-mc-accent/20 text-mc-accent border-2 border-mc-accent shadow-lg shadow-mc-accent/20"
            : "bg-mc-bg-tertiary text-mc-text-secondary border-2 border-mc-border hover:border-mc-accent/50"
        } disabled:opacity-40`}
        aria-label={isMicActive ? "Microphone (active)" : "Microphone (inactive)"}
        disabled={!backendOnline}
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 10v2a7 7 0 01-14 0v-2" />
          <line x1="12" y1="19" x2="12" y2="23" strokeWidth={2} strokeLinecap="round" />
          <line x1="8" y1="23" x2="16" y2="23" strokeWidth={2} strokeLinecap="round" />
        </svg>
      </button>
      <button
        className="p-3 rounded-full bg-mc-bg-tertiary hover:bg-mc-border transition-colors disabled:opacity-40"
        aria-label="Notifications"
        disabled={!backendOnline}
      >
        <svg className="w-5 h-5 text-mc-text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
        </svg>
      </button>
    </div>
  );
}
