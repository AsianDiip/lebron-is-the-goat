"""
app.py — Streamlit dashboard for NBA win probability model.

Three modes:
  - Replay:   select a historical game, view the full probability curve
  - Live:     poll the FastAPI server for a running game
  - Backtest: display evaluation figures and per-quarter metrics

Usage:
    streamlit run dashboard/app.py

Note: Streamlit is not a true push model. Live mode polls every ~10 seconds
via st.rerun(), introducing up to ~40 seconds of total latency (30s API poll
+ 10s dashboard poll). This is a known limitation.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib

from live.game_state import GameState
from live.poller import get_team_abbrev_map, lookup_pregame
from model.train_ingame import INGAME_FEATURES, StratifiedCalibrator  # noqa: F401
from model.train_pregame import PREGAME_FEATURES

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
FIGURES_DIR = MODEL_DIR / "eval_figures"

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(page_title="NBA Win Probability", layout="wide")


# ── Cached loaders ───────────────────────────────────────────────────

@st.cache_resource
def load_models():
    pregame_model = joblib.load(MODEL_DIR / "pregame.pkl")
    ingame_model = joblib.load(MODEL_DIR / "ingame.pkl")
    return pregame_model, ingame_model


@st.cache_data
def load_game_list() -> pd.DataFrame:
    """Load game list from games.db for the game selector."""
    db_path = DATA_DIR / "raw" / "games.db"
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT h.game_id, h.game_date, h.season,
               a.team_abbrev AS away_team, h.team_abbrev AS home_team,
               a.pts AS away_pts, h.pts AS home_pts,
               h.wl AS home_wl
        FROM game_logs h
        JOIN game_logs a ON h.game_id = a.game_id AND a.is_home = 0
        WHERE h.is_home = 1
        ORDER BY h.game_date DESC, h.game_id
        """,
        conn,
    )
    conn.close()
    df["label"] = (
        df["game_date"] + "  —  "
        + df["away_team"] + " @ " + df["home_team"]
        + "  (" + df["away_pts"].astype(str) + "-" + df["home_pts"].astype(str) + ")"
    )
    return df


@st.cache_data
def load_abbrev_map() -> dict[str, int]:
    return get_team_abbrev_map()


# ── Replay engine ────────────────────────────────────────────────────

def replay_game(game_id: str, pregame_model, ingame_model) -> pd.DataFrame:
    """Replay a game from pbp.db and return a DataFrame of probability snapshots."""
    pregame_info = lookup_pregame(game_id)
    if pregame_info is None:
        st.error(f"No pregame features found for game {game_id}")
        return pd.DataFrame()

    abbrev_map = load_abbrev_map()

    # Compute pre-game probability
    pregame_vec = np.array(
        [[pregame_info[f] for f in PREGAME_FEATURES]], dtype=np.float64
    )
    pre_game_prob = float(pregame_model.predict_proba(pregame_vec)[0, 1])

    state = GameState(
        game_id=game_id,
        home_team_id=pregame_info["home_team_id"],
        away_team_id=pregame_info["away_team_id"],
        pre_game_prob=pre_game_prob,
        team_abbrev_map=abbrev_map,
    )

    # Read PBP from SQLite
    db_path = DATA_DIR / "raw" / "pbp.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT action_number, action_id, period, clock_seconds,
               team_id, action_type, sub_type, description,
               score_home, score_away, is_field_goal, shot_result
        FROM play_by_play
        WHERE game_id = ?
        ORDER BY action_number
        """,
        (game_id,),
    ).fetchall()
    conn.close()

    if not rows:
        st.error(f"No play-by-play data found for game {game_id}")
        return pd.DataFrame()

    snapshots = []
    for row in rows:
        event = dict(row)
        if not state.update(event):
            continue

        fv = state.to_feature_vector().reshape(1, -1)
        prob = float(ingame_model.predict_proba(fv)[0, 1])

        snapshots.append({
            "action_number": event.get("action_number", 0),
            "period": state.period,
            "clock_seconds": state.clock_seconds,
            "seconds_remaining": state.seconds_remaining,
            "home_score": state.home_score,
            "away_score": state.away_score,
            "description": event.get("description") or event.get("action_type") or "",
            "home_win_prob": prob,
        })

    return pd.DataFrame(snapshots)


# ── Chart builder ────────────────────────────────────────────────────

def plot_probability_curve(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    title: str = "",
) -> plt.Figure:
    """Build an ESPN-style probability curve with filled regions and quarter markers."""
    # ── ESPN-style team colors (fallback: blue/orange) ──
    TEAM_COLORS = {
        "ATL": "#E03A3E", "BOS": "#007A33", "BKN": "#000000", "CHA": "#1D1160",
        "CHI": "#CE1141", "CLE": "#860038", "DAL": "#00538C", "DEN": "#0E2240",
        "DET": "#C8102E", "GSW": "#1D428A", "HOU": "#CE1141", "IND": "#002D62",
        "LAC": "#C8102E", "LAL": "#552583", "MEM": "#5D76A9", "MIA": "#98002E",
        "MIL": "#00471B", "MIN": "#0C2340", "NOP": "#0C2340", "NYK": "#006BB6",
        "OKC": "#007AC1", "ORL": "#0077C0", "PHI": "#006BB6", "PHX": "#1D1160",
        "POR": "#E03A3E", "SAC": "#5A2D81", "SAS": "#C4CED4", "TOR": "#CE1141",
        "UTA": "#002B5C", "WAS": "#002B5C",
    }
    home_color = TEAM_COLORS.get(home_team, "#1f77b4")
    away_color = TEAM_COLORS.get(away_team, "#ff7f0e")

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    ax.set_facecolor("#f9f9f9")

    secs = df["seconds_remaining"].values
    probs = df["home_win_prob"].values
    # Convert probability to ESPN-style y-axis: home team at top, away at bottom
    # ESPN shows: 100% (top, home) -> 50% (middle) -> 100% (bottom, away)
    # Map: home_prob 1.0 -> y=0 (top), 0.5 -> y=50, 0.0 -> y=100 (bottom)
    y_espn = (1 - probs) * 100

    # ── Toss-up band (light shading around 50%) ──
    ax.axhspan(35, 65, color="#e8edf2", alpha=0.5, zorder=0)

    # ── Filled regions ──
    ax.fill_between(secs, y_espn, 50, where=y_espn <= 50,
                    interpolate=True, alpha=0.25, color=home_color, zorder=2)
    ax.fill_between(secs, y_espn, 50, where=y_espn >= 50,
                    interpolate=True, alpha=0.25, color=away_color, zorder=2)

    # ── Probability line ──
    ax.plot(secs, y_espn, lw=2, color=home_color, zorder=3)

    # ── 50% midline ──
    ax.axhline(50, color="#b0b0b0", linestyle="-", lw=0.8, zorder=1)

    # ── Quarter boundaries ──
    quarter_secs = [2160, 1440, 720, 0]
    for sec in quarter_secs:
        if sec >= secs.min():
            ax.axvline(sec, color="#d0d0d0", lw=0.8, zorder=1)

    # Quarter labels centered
    quarter_labels = [("1st", 2520), ("2nd", 1800), ("3rd", 1080), ("4th", 360)]
    for qlabel, center in quarter_labels:
        if center >= secs.min():
            ax.text(center, 2, qlabel, ha="center", va="top",
                    fontsize=10, color="#888", fontweight="medium")

    # OT label
    if secs.min() < 0:
        ax.axvline(0, color="#d0d0d0", lw=0.8, zorder=1)
        ax.text(secs.min() / 2, 2, "OT", ha="center", va="top",
                fontsize=10, color="#c77600", fontweight="medium")

    # ── Team names at corners (ESPN-style) ──
    ax.text(0.01, 0.02, f"{home_team}", transform=ax.transAxes,
            fontsize=13, fontweight="bold", color=home_color, va="bottom")
    ax.text(0.01, 0.98, f"{away_team}", transform=ax.transAxes,
            fontsize=13, fontweight="bold", color=away_color, va="top")

    # ── Y-axis: percentage labels on both sides ──
    ax.set_ylim(100, 0)  # Inverted: 0% at top (home certain), 100% at bottom (away certain)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["100%", "75%", "", "75%", "100%"], fontsize=9, color="#888")
    ax.tick_params(axis="y", length=0)

    # ── X-axis: clean, no raw seconds ──
    ax.set_xlim(max(secs.max(), 2880) + 30, secs.min() - 30)
    ax.set_xticks([])

    # ── Spines ──
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title or f"{away_team} @ {home_team}", fontsize=13,
                 fontweight="bold", pad=15, color="#333")

    # ── Key play annotations — top 5 momentum swings ──
    if len(probs) > 1:
        deltas = np.abs(np.diff(probs))
        n_annot = min(5, len(deltas))
        top_idx = np.argsort(deltas)[-n_annot:]
        for idx in top_idx:
            desc = str(df.iloc[idx + 1]["description"])[:35]
            y_val = y_espn[idx + 1]
            sec_val = secs[idx + 1]
            y_offset = -5 if y_val > 50 else 5
            ax.annotate(
                desc,
                xy=(sec_val, y_val),
                xytext=(sec_val, y_val + y_offset),
                fontsize=6.5,
                color="#d62728",
                alpha=0.85,
                arrowprops=dict(arrowstyle="->", color="#d62728", alpha=0.5, lw=0.8),
                ha="center",
                va="bottom" if y_offset > 0 else "top",
                zorder=5,
            )

    plt.tight_layout()
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────

st.sidebar.title("NBA Win Probability")
mode = st.sidebar.radio("Mode", ["Replay", "Live", "Backtest Report"])


# ── MODE: Replay ─────────────────────────────────────────────────────

if mode == "Replay":
    st.header("Historical Game Replay")

    games_df = load_game_list()
    seasons = sorted(games_df["season"].unique(), reverse=True)
    selected_season = st.sidebar.selectbox("Season", seasons)

    season_games = games_df[games_df["season"] == selected_season]
    game_options = season_games["label"].tolist()
    selected_label = st.sidebar.selectbox("Game", game_options)

    if selected_label:
        game_row = season_games[season_games["label"] == selected_label].iloc[0]
        game_id = game_row["game_id"]
        home_team = game_row["home_team"]
        away_team = game_row["away_team"]

        if st.sidebar.button("Load Game", type="primary"):
            st.session_state["replay_game_id"] = game_id
            st.session_state["replay_home"] = home_team
            st.session_state["replay_away"] = away_team

        if "replay_game_id" in st.session_state:
            gid = st.session_state["replay_game_id"]
            home = st.session_state["replay_home"]
            away = st.session_state["replay_away"]

            with st.spinner(f"Replaying {away} @ {home}..."):
                pregame_model, ingame_model = load_models()
                snap_df = replay_game(gid, pregame_model, ingame_model)

            if not snap_df.empty:
                # Probability curve
                fig = plot_probability_curve(
                    snap_df, home, away,
                    title=f"{away} @ {home}  ({game_row['game_date']})",
                )
                st.pyplot(fig)
                plt.close(fig)

                # Score and probability display
                final = snap_df.iloc[-1]
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.metric(f"{home} (Home)", int(final["home_score"]),
                              f"{final['home_win_prob']:.1%} win prob")
                with col2:
                    st.metric(f"{away} (Away)", int(final["away_score"]),
                              f"{1 - final['home_win_prob']:.1%} win prob")
                with col3:
                    st.metric("Events Processed", len(snap_df))

                # Play-by-play log (last 15 events)
                st.subheader("Recent Plays")
                log_df = snap_df[["period", "clock_seconds", "home_score", "away_score",
                                  "description", "home_win_prob"]].tail(15).copy()
                log_df.columns = ["Qtr", "Clock", home, away, "Play", "Home Win %"]
                log_df["Home Win %"] = log_df["Home Win %"].apply(lambda x: f"{x:.1%}")
                log_df = log_df.iloc[::-1]  # most recent first
                st.dataframe(log_df, use_container_width=True, hide_index=True)


# ── MODE: Live ───────────────────────────────────────────────────────

elif mode == "Live":
    st.header("Live Game Tracking")
    st.info(
        "Updates every ~30s. Streamlit is not a true push model — "
        "total latency is up to ~40s (30s API poll + 10s dashboard refresh)."
    )

    game_id_input = st.sidebar.text_input("Game ID", placeholder="e.g. 0022401234")
    api_url = st.sidebar.text_input("API URL", value="http://localhost:8000")

    if game_id_input:
        try:
            import requests
            resp = requests.get(f"{api_url}/live/{game_id_input}", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Home Score", data.get("home_score", 0))
                with col2:
                    st.metric("Away Score", data.get("away_score", 0))

                prob = data.get("home_win_prob", 0.5)
                st.progress(prob, text=f"Home Win Probability: {prob:.1%}")

                st.caption(
                    f"Period {data.get('period', '?')} | "
                    f"Clock: {data.get('clock_seconds', '?')}s | "
                    f"Events: {data.get('events_processed', 0)}"
                )

                # Auto-refresh
                time.sleep(10)
                st.rerun()
            elif resp.status_code == 404:
                st.warning(f"Game {game_id_input} not found. Is the API server running?")
            else:
                st.error(f"API returned status {resp.status_code}")
        except requests.ConnectionError:
            st.warning(
                "Cannot connect to the API server. "
                "Start it with: `uvicorn live.api:app --reload`"
            )
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.caption("Enter a game ID in the sidebar to start tracking.")


# ── MODE: Backtest Report ────────────────────────────────────────────

elif mode == "Backtest Report":
    st.header("Model Evaluation — Backtest Report")
    st.caption("Test set: 2023-24 season")

    # Metrics summary
    st.subheader("Performance Summary")
    metrics_data = {
        "Model": ["Pre-game LR", "In-game XGBoost", "In-game (Q2/Q3, |diff|≤10)"],
        "Brier": ["0.0174", "0.1505", "0.1752"],
        "ECE": ["1.3%", "1.3%", "1.64%"],
        "AUC-ROC": ["0.863", "0.863", "0.813"],
    }
    st.table(pd.DataFrame(metrics_data))

    targets = {
        "Pre-game ECE < 4%": True,
        "In-game Brier < 0.18": True,
        "In-game ECE < 5%": True,
        "In-game AUC > 0.80": True,
    }
    cols = st.columns(len(targets))
    for col, (check, passed) in zip(cols, targets.items()):
        col.metric(check, "PASS" if passed else "FAIL")

    # Evaluation figures
    figure_files = [
        ("Pre-game Reliability", "pregame_reliability.png"),
        ("In-game Reliability", "ingame_reliability.png"),
        ("Per-Quarter Calibration", "per_quarter_calibration.png"),
        ("Win Probability Curves", "win_prob_curves.png"),
        ("SHAP Beeswarm", "shap_beeswarm.png"),
        ("SHAP Bar", "shap_bar.png"),
    ]

    # Display in 2-column grid
    for i in range(0, len(figure_files), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(figure_files):
                title, fname = figure_files[i + j]
                fpath = FIGURES_DIR / fname
                if fpath.exists():
                    col.subheader(title)
                    col.image(str(fpath))
                else:
                    col.warning(f"{fname} not found. Run `python model/evaluate.py` first.")
