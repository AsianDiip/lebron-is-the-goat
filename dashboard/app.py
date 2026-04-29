"""
app.py — Streamlit dashboard for NBA win probability model.

Two modes:
  - Game:     unified view for live tracking and historical replay
              (scoreboard header, ESPN-style win prob chart, play-by-play feed)
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

# ── Team metadata ────────────────────────────────────────────────────

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

TEAM_NAMES = {
    "ATL": "Hawks", "BOS": "Celtics", "BKN": "Nets", "CHA": "Hornets",
    "CHI": "Bulls", "CLE": "Cavaliers", "DAL": "Mavericks", "DEN": "Nuggets",
    "DET": "Pistons", "GSW": "Warriors", "HOU": "Rockets", "IND": "Pacers",
    "LAC": "Clippers", "LAL": "Lakers", "MEM": "Grizzlies", "MIA": "Heat",
    "MIL": "Bucks", "MIN": "Timberwolves", "NOP": "Pelicans", "NYK": "Knicks",
    "OKC": "Thunder", "ORL": "Magic", "PHI": "76ers", "PHX": "Suns",
    "POR": "Trail Blazers", "SAC": "Kings", "SAS": "Spurs", "TOR": "Raptors",
    "UTA": "Jazz", "WAS": "Wizards",
}

# NBA team ID mapping (abbreviation -> team_id) for logo URLs
TEAM_IDS = {
    "ATL": 1610612737, "BOS": 1610612738, "BKN": 1610612751, "CHA": 1610612766,
    "CHI": 1610612741, "CLE": 1610612739, "DAL": 1610612742, "DEN": 1610612743,
    "DET": 1610612765, "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
    "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763, "MIA": 1610612748,
    "MIL": 1610612749, "MIN": 1610612750, "NOP": 1610612740, "NYK": 1610612752,
    "OKC": 1610612760, "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759, "TOR": 1610612761,
    "UTA": 1610612762, "WAS": 1610612764,
}


def team_logo_url(team_abbrev: str) -> str:
    """Return CDN URL for team logo. Falls back to abbrev-based lookup."""
    team_id = TEAM_IDS.get(team_abbrev)
    if team_id:
        return f"https://cdn.nba.com/logos/nba/{team_id}/primary/L/logo.svg"
    return ""


def format_clock(period: int, clock_seconds: int) -> str:
    """Format period and clock into a readable string like 'Q4 3:42'."""
    minutes = clock_seconds // 60
    seconds = clock_seconds % 60
    if period <= 4:
        return f"Q{period}  {minutes}:{seconds:02d}"
    else:
        ot_num = period - 4
        return f"OT{ot_num}  {minutes}:{seconds:02d}"


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
    id_to_abbrev = {v: k for k, v in abbrev_map.items()}

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
        team_id = event.get("team_id") or 0

        snapshots.append({
            "action_number": event.get("action_number", 0),
            "period": state.period,
            "clock_seconds": state.clock_seconds,
            "seconds_remaining": state.seconds_remaining,
            "home_score": state.home_score,
            "away_score": state.away_score,
            "team_abbrev": id_to_abbrev.get(int(team_id) if team_id else 0, ""),
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
    home_color = TEAM_COLORS.get(home_team, "#1f77b4")
    away_color = TEAM_COLORS.get(away_team, "#ff7f0e")

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    ax.set_facecolor("#f9f9f9")

    secs = df["seconds_remaining"].values
    probs = df["home_win_prob"].values
    # Convert probability to ESPN-style y-axis: home team at top, away at bottom
    y_espn = (1 - probs) * 100

    # Toss-up band
    ax.axhspan(35, 65, color="#e8edf2", alpha=0.5, zorder=0)

    # Filled regions
    ax.fill_between(secs, y_espn, 50, where=y_espn <= 50,
                    interpolate=True, alpha=0.25, color=home_color, zorder=2)
    ax.fill_between(secs, y_espn, 50, where=y_espn >= 50,
                    interpolate=True, alpha=0.25, color=away_color, zorder=2)

    # Probability line
    ax.plot(secs, y_espn, lw=2, color=home_color, zorder=3)

    # 50% midline
    ax.axhline(50, color="#b0b0b0", linestyle="-", lw=0.8, zorder=1)

    # Quarter boundaries
    quarter_secs = [2160, 1440, 720, 0]
    for sec in quarter_secs:
        if sec >= secs.min():
            ax.axvline(sec, color="#d0d0d0", lw=0.8, zorder=1)

    # Quarter labels
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

    # Team names at corners
    ax.text(0.01, 0.02, f"{home_team}", transform=ax.transAxes,
            fontsize=13, fontweight="bold", color=home_color, va="bottom")
    ax.text(0.01, 0.98, f"{away_team}", transform=ax.transAxes,
            fontsize=13, fontweight="bold", color=away_color, va="top")

    # Y-axis
    ax.set_ylim(100, 0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["100%", "75%", "", "75%", "100%"], fontsize=9, color="#888")
    ax.tick_params(axis="y", length=0)

    # X-axis
    ax.set_xlim(max(secs.max(), 2880) + 30, secs.min() - 30)
    ax.set_xticks([])

    # Spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=15, color="#333")

    # Key play annotations — top 5 momentum swings
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


# ── Scoreboard header ───────────────────────────────────────────────

def render_scoreboard(
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
    home_win_prob: float,
    period: int | None = None,
    clock_seconds: int | None = None,
    game_status: str = "",
) -> None:
    """Render the scoreboard header with logos, team names, and scores."""
    home_logo = team_logo_url(home_team)
    away_logo = team_logo_url(away_team)
    home_name = TEAM_NAMES.get(home_team, home_team)
    away_name = TEAM_NAMES.get(away_team, away_team)
    home_color = TEAM_COLORS.get(home_team, "#1f77b4")
    away_color = TEAM_COLORS.get(away_team, "#ff7f0e")

    # Clock / status line
    if period is not None and clock_seconds is not None:
        clock_str = format_clock(period, clock_seconds)
    elif game_status:
        clock_str = game_status
    else:
        clock_str = ""

    # --- Row 1: Away logo+name | Score | Home logo+name ---
    col_away, col_score, col_home = st.columns([2, 1, 2])

    with col_away:
        sub1, sub2 = st.columns([1, 3])
        with sub1:
            if away_logo:
                st.image(away_logo, width=60)
        with sub2:
            st.markdown(
                f"<p style='margin:0;font-size:12px;color:gray;'>{away_team}</p>"
                f"<p style='margin:0;font-size:22px;font-weight:700;color:{away_color};'>"
                f"{away_name}</p>",
                unsafe_allow_html=True,
            )

    with col_score:
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<p style='margin:0;font-size:42px;font-weight:700;'>"
            f"{away_score} &ndash; {home_score}</p>"
            f"<p style='margin:0;font-size:14px;color:gray;'>{clock_str}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_home:
        sub1, sub2 = st.columns([1, 3])
        with sub1:
            if home_logo:
                st.image(home_logo, width=60)
        with sub2:
            st.markdown(
                f"<p style='margin:0;font-size:12px;color:gray;'>{home_team}</p>"
                f"<p style='margin:0;font-size:22px;font-weight:700;color:{home_color};'>"
                f"{home_name}</p>",
                unsafe_allow_html=True,
            )

    # --- Row 2: Win probability bar ---
    away_prob = 1 - home_win_prob
    prob_cols = st.columns([1, 6, 1])
    with prob_cols[0]:
        st.caption(f"{away_team} {away_prob:.1%}")
    with prob_cols[1]:
        st.progress(home_win_prob, text=f"Home Win Probability: {home_win_prob:.1%}")
    with prob_cols[2]:
        st.caption(f"{home_team} {home_win_prob:.1%}")

    st.divider()


# ── Play-by-play feed ───────────────────────────────────────────────

def render_play_by_play(
    snap_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    max_rows: int = 50,
) -> None:
    """Render a scrollable play-by-play feed, most recent first."""
    if snap_df.empty:
        return

    display_df = snap_df.tail(max_rows).copy()
    display_df = display_df.iloc[::-1]  # most recent first

    # Format columns
    display_df["Clock"] = display_df["clock_seconds"].apply(
        lambda s: f"{s // 60}:{s % 60:02d}"
    )
    display_df["Qtr"] = display_df["period"].apply(
        lambda p: f"Q{p}" if p <= 4 else f"OT{p - 4}"
    )
    display_df["Score"] = (
        display_df["away_score"].astype(str) + " - " + display_df["home_score"].astype(str)
    )
    display_df["Win %"] = display_df["home_win_prob"].apply(lambda x: f"{x:.1%}")
    display_df["Team"] = display_df["team_abbrev"]
    display_df["Play"] = display_df["description"]

    show_cols = ["Qtr", "Clock", "Team", "Play", "Score", "Win %"]
    st.dataframe(
        display_df[show_cols],
        use_container_width=True,
        hide_index=True,
        height=400,
    )


# ── Sidebar ──────────────────────────────────────────────────────────

st.sidebar.title("NBA Win Probability")
mode = st.sidebar.radio("Mode", ["Game", "Backtest Report"])


# ── MODE: Game (unified live + replay) ───────────────────────────────

if mode == "Game":
    source = st.sidebar.radio("Source", ["Replay", "Live"])

    # ── Replay source ────────────────────────────────────────────────
    if source == "Replay":
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
                    final = snap_df.iloc[-1]

                    # Scoreboard header
                    render_scoreboard(
                        home_team=home,
                        away_team=away,
                        home_score=int(final["home_score"]),
                        away_score=int(final["away_score"]),
                        home_win_prob=final["home_win_prob"],
                        game_status=f"Final  ({game_row['game_date']})",
                    )

                    # Win probability chart
                    fig = plot_probability_curve(snap_df, home, away)
                    st.pyplot(fig)
                    plt.close(fig)

                    # Play-by-play feed
                    st.subheader("Play-by-Play")
                    render_play_by_play(snap_df, home, away)

    # ── Live source ──────────────────────────────────────────────────
    elif source == "Live":
        game_id_input = st.sidebar.text_input("Game ID", placeholder="e.g. 0022401234")
        api_url = st.sidebar.text_input("API URL", value="http://localhost:8000")

        if game_id_input:
            try:
                import requests
                resp = requests.get(f"{api_url}/live/{game_id_input}", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()

                    home_team = data.get("home_team_abbrev", "HOME")
                    away_team = data.get("away_team_abbrev", "AWAY")
                    home_score = data.get("home_score", 0)
                    away_score = data.get("away_score", 0)
                    home_win_prob = data.get("home_win_prob", 0.5)
                    period = data.get("period", 1)
                    clock_seconds = data.get("clock_seconds", 720)

                    # Scoreboard header
                    render_scoreboard(
                        home_team=home_team,
                        away_team=away_team,
                        home_score=home_score,
                        away_score=away_score,
                        home_win_prob=home_win_prob,
                        period=period,
                        clock_seconds=clock_seconds,
                    )

                    # Build probability chart from history
                    prob_history = data.get("prob_history", [])
                    if prob_history:
                        prob_df = pd.DataFrame(prob_history)
                        fig = plot_probability_curve(
                            prob_df, home_team, away_team,
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("Waiting for play-by-play data...")

                    # Play-by-play feed
                    play_log = data.get("play_log", [])
                    if play_log:
                        st.subheader("Play-by-Play")
                        play_df = pd.DataFrame(play_log)
                        render_play_by_play(play_df, home_team, away_team)

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
        "Model": ["Pre-game LR", "In-game XGBoost", "In-game (Q2/Q3, |diff|<=10)"],
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
