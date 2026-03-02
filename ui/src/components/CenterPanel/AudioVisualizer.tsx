"use client";

import { useEffect, useRef } from "react";

type Props = {
  status?: string; // "listening" | "speaking" | "idle"
  active?: boolean;
};

export default function AudioVisualizer({ status = "idle", active = false }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const barsRef = useRef([0.3, 0.5, 0.7, 0.4, 0.6]);
  const statusRef = useRef(status);
  const activeRef = useRef(active);
  statusRef.current = status;
  activeRef.current = active;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const size = 200;
    canvas.width = size;
    canvas.height = size;

    let frameId: number;
    const animate = () => {
      ctx.clearRect(0, 0, size, size);
      const cx = size / 2;
      const cy = size / 2;

      const isActive = activeRef.current;
      const curStatus = statusRef.current;
      const isSpeaking = curStatus === "speaking";

      // Color based on status
      const ringColor = !isActive
        ? "rgba(239, 68, 68, 0.15)"
        : isSpeaking
        ? "rgba(250, 204, 21, 0.2)"
        : "rgba(88, 166, 255, 0.2)";
      const barColor = !isActive
        ? "#ef4444"
        : isSpeaking
        ? "#facc15"
        : "#58a6ff";
      const glowColor = !isActive
        ? "rgba(239, 68, 68, 0.1)"
        : isSpeaking
        ? "rgba(250, 204, 21, 0.1)"
        : "rgba(88, 166, 255, 0.08)";

      // Outer glow
      const gradient = ctx.createRadialGradient(cx, cy, 40, cx, cy, 100);
      gradient.addColorStop(0, glowColor);
      gradient.addColorStop(1, "transparent");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, size, size);

      // Concentric rings
      for (let r = 30; r <= 90; r += 15) {
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.strokeStyle = ringColor;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Inner bars
      const barCount = 5;
      const barWidth = 8;
      const barGap = 4;
      const totalWidth = barCount * barWidth + (barCount - 1) * barGap;
      const startX = cx - totalWidth / 2 + barWidth / 2;

      const intensity = isSpeaking ? 0.35 : isActive ? 0.2 : 0.05;
      const bars = barsRef.current;
      for (let i = 0; i < bars.length; i++) {
        const delta = (Math.random() - 0.5) * intensity;
        bars[i] = Math.max(0.15, Math.min(1, bars[i] + delta));
      }

      const maxBarHeight = isSpeaking ? 35 : isActive ? 25 : 12;
      bars.forEach((h, i) => {
        const x = startX + i * (barWidth + barGap);
        const barHeight = maxBarHeight * h;
        ctx.fillStyle = barColor;
        ctx.globalAlpha = 0.9;
        ctx.fillRect(x - barWidth / 2, cy - barHeight / 2, barWidth, barHeight);
        ctx.globalAlpha = 1;
      });

      frameId = requestAnimationFrame(animate);
    };

    frameId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frameId);
  }, []);

  return (
    <div className="flex items-center justify-center">
      <canvas
        ref={canvasRef}
        className="w-48 h-48"
        style={{ width: 200, height: 200 }}
        width={200}
        height={200}
      />
    </div>
  );
}
