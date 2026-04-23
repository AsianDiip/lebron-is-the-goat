"""
fetch_games.py — Fetch season-level game logs and team efficiency stats.

Fetches:
  - LeagueGameLog  → games.db / game_logs
  - LeagueDashTeamStats (Advanced) → games.db / team_efficiency

Run:
    python data/fetch_games.py [--season 2023-24]
"""

import argparse
import sqlite3
import time
from pathlib import Path

from nba_api.stats.endpoints import LeagueGameLog, LeagueDashTeamStats

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
DB_PATH = Path(__file__).parent / "raw" / "games.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def setup_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS game_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            season      TEXT    NOT NULL,
            game_id     TEXT    NOT NULL,
            game_date   TEXT    NOT NULL,
            team_id     INTEGER NOT NULL,
            team_abbrev TEXT    NOT NULL,
            matchup     TEXT    NOT NULL,
            is_home     INTEGER NOT NULL,
            wl          TEXT    NOT NULL,
            pts         INTEGER,
            plus_minus  INTEGER,
            UNIQUE(game_id, team_id)
        );
        CREATE INDEX IF NOT EXISTS idx_game_logs_game_id ON game_logs(game_id);
        CREATE INDEX IF NOT EXISTS idx_game_logs_date    ON game_logs(game_date);
        CREATE INDEX IF NOT EXISTS idx_game_logs_team_id ON game_logs(team_id);
        CREATE INDEX IF NOT EXISTS idx_game_logs_season  ON game_logs(season);

        CREATE TABLE IF NOT EXISTS team_efficiency (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            season      TEXT    NOT NULL,
            team_id     INTEGER NOT NULL,
            team_name   TEXT    NOT NULL,
            gp          INTEGER,
            w           INTEGER,
            l           INTEGER,
            w_pct       REAL,
            off_rating  REAL,
            def_rating  REAL,
            net_rating  REAL,
            pace        REAL,
            efg_pct     REAL,
            ts_pct      REAL,
            tm_tov_pct  REAL,
            oreb_pct    REAL,
            UNIQUE(season, team_id)
        );
        CREATE INDEX IF NOT EXISTS idx_team_eff_team_id ON team_efficiency(team_id);
        CREATE INDEX IF NOT EXISTS idx_team_eff_season  ON team_efficiency(season);
    """)
    conn.commit()


def fetch_game_logs(conn: sqlite3.Connection, season: str) -> None:
    count = conn.execute(
        "SELECT COUNT(*) FROM game_logs WHERE season = ?", (season,)
    ).fetchone()[0]
    if count > 0:
        print(f"  game_logs [{season}]: already cached ({count} rows), skipping.")
        return

    print(f"  game_logs [{season}]: fetching from API...")
    df = LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]

    rows = []
    for _, row in df.iterrows():
        matchup = row["MATCHUP"]
        is_home = 1 if " vs. " in matchup else 0
        rows.append((
            season,
            row["GAME_ID"],
            row["GAME_DATE"],
            int(row["TEAM_ID"]),
            row["TEAM_ABBREVIATION"],
            matchup,
            is_home,
            row["WL"],
            int(row["PTS"]) if row["PTS"] == row["PTS"] else None,
            int(row["PLUS_MINUS"]) if row["PLUS_MINUS"] == row["PLUS_MINUS"] else None,
        ))

    conn.executemany(
        """INSERT OR IGNORE INTO game_logs
           (season, game_id, game_date, team_id, team_abbrev, matchup, is_home, wl, pts, plus_minus)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    print(f"  game_logs [{season}]: inserted {len(rows)} rows.")


def fetch_team_efficiency(conn: sqlite3.Connection, season: str) -> None:
    count = conn.execute(
        "SELECT COUNT(*) FROM team_efficiency WHERE season = ?", (season,)
    ).fetchone()[0]
    if count > 0:
        print(f"  team_efficiency [{season}]: already cached ({count} rows), skipping.")
        return

    print(f"  team_efficiency [{season}]: fetching from API...")
    df = LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]

    required_cols = ["OFF_RATING", "DEF_RATING", "NET_RATING", "PACE", "EFG_PCT", "TS_PCT", "TM_TOV_PCT", "OREB_PCT"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"LeagueDashTeamStats Advanced missing expected columns: {missing}. "
            "API schema may have changed."
        )

    rows = []
    for _, row in df.iterrows():
        rows.append((
            season,
            int(row["TEAM_ID"]),
            row["TEAM_NAME"],
            int(row["GP"]),
            int(row["W"]),
            int(row["L"]),
            float(row["W_PCT"]),
            float(row["OFF_RATING"]),
            float(row["DEF_RATING"]),
            float(row["NET_RATING"]),
            float(row["PACE"]),
            float(row["EFG_PCT"]),
            float(row["TS_PCT"]),
            float(row["TM_TOV_PCT"]),
            float(row["OREB_PCT"]),
        ))

    conn.executemany(
        """INSERT OR IGNORE INTO team_efficiency
           (season, team_id, team_name, gp, w, l, w_pct,
            off_rating, def_rating, net_rating, pace, efg_pct, ts_pct, tm_tov_pct, oreb_pct)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    print(f"  team_efficiency [{season}]: inserted {len(rows)} rows.")


def run(seasons: list[str]) -> None:
    conn = get_connection()
    setup_schema(conn)

    for season in seasons:
        print(f"\n--- Season: {season} ---")
        fetch_game_logs(conn, season)
        time.sleep(0.6)
        fetch_team_efficiency(conn, season)
        time.sleep(0.6)

    conn.close()
    print("\nDone. games.db is up to date.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NBA game logs and team efficiency stats.")
    parser.add_argument("--season", help="Fetch a single season (e.g. 2023-24). Defaults to all seasons.")
    args = parser.parse_args()

    seasons_to_fetch = [args.season] if args.season else SEASONS
    if args.season and args.season not in SEASONS:
        raise ValueError(f"Unknown season '{args.season}'. Valid options: {SEASONS}")

    run(seasons_to_fetch)
