export function formatClock(period: number, clockSeconds: number): string {
  const minutes = Math.floor(clockSeconds / 60);
  const seconds = clockSeconds % 60;
  const pad = seconds < 10 ? "0" : "";
  if (period <= 4) {
    return `Q${period}  ${minutes}:${pad}${seconds}`;
  }
  return `OT${period - 4}  ${minutes}:${pad}${seconds}`;
}

export function formatPeriod(period: number): string {
  if (period <= 4) return `Q${period}`;
  return `OT${period - 4}`;
}

export function formatPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatWinProb(prob: number): string {
  return `${(prob * 100).toFixed(1)}%`;
}
