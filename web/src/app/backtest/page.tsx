"use client";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const METRICS = [
  { model: "Pre-game LR", brier: "0.0174", ece: "1.3%", auc: "0.863" },
  { model: "In-game XGBoost", brier: "0.1505", ece: "1.3%", auc: "0.863" },
  { model: "In-game (Q2/Q3, |diff|<=10)", brier: "0.1752", ece: "1.64%", auc: "0.813" },
];

const TARGETS = [
  { label: "Pre-game ECE < 4%", passed: true },
  { label: "In-game Brier < 0.18", passed: true },
  { label: "In-game ECE < 5%", passed: true },
  { label: "In-game AUC > 0.80", passed: true },
];

const FIGURES = [
  { title: "Pre-game Reliability", file: "pregame_reliability.png" },
  { title: "In-game Reliability", file: "ingame_reliability.png" },
  { title: "Per-Quarter Calibration", file: "per_quarter_calibration.png" },
  { title: "Win Probability Curves", file: "win_prob_curves.png" },
  { title: "SHAP Beeswarm", file: "shap_beeswarm.png" },
  { title: "SHAP Bar", file: "shap_bar.png" },
];

export default function BacktestPage() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">Model Evaluation</h1>
        <p className="text-sm text-gray-400 mt-1">Test set: 2023-24 season</p>
      </div>

      {/* Metrics table */}
      <div className="rounded-xl bg-white/[0.03] border border-white/10 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs uppercase tracking-wider text-gray-400 border-b border-white/10">
              <th className="py-3 px-4 text-left">Model</th>
              <th className="py-3 px-4 text-center">Brier</th>
              <th className="py-3 px-4 text-center">ECE</th>
              <th className="py-3 px-4 text-center">AUC-ROC</th>
            </tr>
          </thead>
          <tbody>
            {METRICS.map((m) => (
              <tr key={m.model} className="border-b border-white/[0.03]">
                <td className="py-3 px-4 text-gray-200 font-medium">{m.model}</td>
                <td className="py-3 px-4 text-center text-gray-300 font-mono">{m.brier}</td>
                <td className="py-3 px-4 text-center text-gray-300 font-mono">{m.ece}</td>
                <td className="py-3 px-4 text-center text-gray-300 font-mono">{m.auc}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pass/fail badges */}
      <div className="grid grid-cols-4 gap-3">
        {TARGETS.map((t) => (
          <div
            key={t.label}
            className={`rounded-lg border p-3 text-center ${
              t.passed
                ? "bg-green-500/5 border-green-500/20"
                : "bg-red-500/5 border-red-500/20"
            }`}
          >
            <p className={`text-lg font-bold ${t.passed ? "text-green-400" : "text-red-400"}`}>
              {t.passed ? "PASS" : "FAIL"}
            </p>
            <p className="text-xs text-gray-400 mt-1">{t.label}</p>
          </div>
        ))}
      </div>

      {/* Evaluation figures */}
      <div className="grid grid-cols-2 gap-4">
        {FIGURES.map((fig) => (
          <div
            key={fig.file}
            className="rounded-xl bg-white/[0.03] border border-white/10 overflow-hidden"
          >
            <div className="px-4 py-3 border-b border-white/5">
              <h3 className="text-sm font-semibold text-gray-300">{fig.title}</h3>
            </div>
            <div className="p-2">
              <img
                src={`${API_BASE}/figures/${fig.file}`}
                alt={fig.title}
                className="w-full rounded"
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = "none";
                  (e.target as HTMLImageElement).parentElement!.innerHTML =
                    '<p class="text-center text-gray-500 py-8 text-sm">Figure not found. Run python model/evaluate.py</p>';
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
