"""
fetch_players.py — Fetch player box scores for all historical games.

Fetches BoxScoreTraditionalV3 for each game_id in games.db → players.db / player_box_scores.

Run:
    python data/fetch_players.py [--season 2023-24] [--resume]

--resume: only attempt game_ids listed in data/raw/failed_players.txt
"""

import argparse
import sqlite3
import time
from pathlib import Path

import requests
from nba_api.stats.endpoints import BoxScoreTraditionalV3

SEASONS = [
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
]
GAMES_DB = Path(__file__).parent / "raw" / "games.db"
PLAYERS_DB = Path(__file__).parent / "raw" / "players.db"
FAILED_FILE = Path(__file__).parent / "raw" / "failed_players.txt"


def get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def setup_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS player_box_scores (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id           TEXT    NOT NULL,
            season            TEXT    NOT NULL,
            team_id           INTEGER NOT NULL,
            team_abbreviation TEXT    NOT NULL,
            player_id         INTEGER NOT NULL,
            player_name       TEXT    NOT NULL,
            start_position    TEXT,
            minutes           TEXT,
            fgm               INTEGER,
            fga               INTEGER,
            fg_pct            REAL,
            fg3m              INTEGER,
            fg3a              INTEGER,
            fg3_pct           REAL,
            ftm               INTEGER,
            fta               INTEGER,
            ft_pct            REAL,
            oreb              INTEGER,
            dreb              INTEGER,
            reb               INTEGER,
            ast               INTEGER,
            stl               INTEGER,
            blk               INTEGER,
            tov               INTEGER,
            pf                INTEGER,
            pts               INTEGER,
            plus_minus        REAL,
            UNIQUE(game_id, player_id)
        );
        CREATE INDEX IF NOT EXISTS idx_player_box_game_id   ON player_box_scores(game_id);
        CREATE INDEX IF NOT EXISTS idx_player_box_team_id   ON player_box_scores(team_id);
        CREATE INDEX IF NOT EXISTS idx_player_box_player_id ON player_box_scores(player_id);
        CREATE INDEX IF NOT EXISTS idx_player_box_season    ON player_box_scores(season);
    """)
    conn.commit()


def get_game_ids(seasons: list[str]) -> list[tuple[str, str]]:
    if not GAMES_DB.exists():
        raise RuntimeError(
            f"games.db not found at {GAMES_DB}. Run fetch_games.py first."
        )
    conn = sqlite3.connect(GAMES_DB)
    placeholders = ",".join("?" * len(seasons))
    rows = conn.execute(
        f"SELECT DISTINCT game_id, season FROM game_logs WHERE season IN ({placeholders})",
        seasons,
    ).fetchall()
    conn.close()
    if not rows:
        raise RuntimeError(
            f"No game_ids found for seasons {seasons}. Run fetch_games.py first."
        )
    return rows


def is_cached(conn: sqlite3.Connection, game_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM player_box_scores WHERE game_id = ? LIMIT 1", (game_id,)
    ).fetchone() is not None


def _safe_int(val) -> int | None:
    if val is None or val != val:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _safe_float(val) -> float | None:
    if val is None or val != val:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def fetch_game_box_scores(conn: sqlite3.Connection, game_id: str, season: str) -> None:
    endpoint = BoxScoreTraditionalV3(game_id=game_id)
    df = endpoint.player_stats.get_data_frame()

    rows = []
    for _, row in df.iterrows():
        first = str(row.get("firstName", "") or "")
        last = str(row.get("familyName", "") or "")
        player_name = f"{first} {last}".strip()

        rows.append((
            game_id,
            season,
            _safe_int(row.get("teamId")),
            str(row.get("teamTricode", "") or ""),
            _safe_int(row.get("personId")),
            player_name,
            str(row.get("position", "") or ""),
            str(row.get("minutes", "") or ""),
            _safe_int(row.get("fieldGoalsMade")),
            _safe_int(row.get("fieldGoalsAttempted")),
            _safe_float(row.get("fieldGoalsPercentage")),
            _safe_int(row.get("threePointersMade")),
            _safe_int(row.get("threePointersAttempted")),
            _safe_float(row.get("threePointersPercentage")),
            _safe_int(row.get("freeThrowsMade")),
            _safe_int(row.get("freeThrowsAttempted")),
            _safe_float(row.get("freeThrowsPercentage")),
            _safe_int(row.get("reboundsOffensive")),
            _safe_int(row.get("reboundsDefensive")),
            _safe_int(row.get("reboundsTotal")),
            _safe_int(row.get("assists")),
            _safe_int(row.get("steals")),
            _safe_int(row.get("blocks")),
            _safe_int(row.get("turnovers")),
            _safe_int(row.get("foulsPersonal")),
            _safe_int(row.get("points")),
            _safe_float(row.get("plusMinusPoints")),
        ))

    conn.execute("BEGIN")
    conn.executemany(
        """INSERT OR IGNORE INTO player_box_scores
           (game_id, season, team_id, team_abbreviation, player_id, player_name,
            start_position, minutes,
            fgm, fga, fg_pct, fg3m, fg3a, fg3_pct, ftm, fta, ft_pct,
            oreb, dreb, reb, ast, stl, blk, tov, pf, pts, plus_minus)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.execute("COMMIT")


def load_failed() -> set[str]:
    if not FAILED_FILE.exists():
        return set()
    return set(line.strip() for line in FAILED_FILE.read_text().splitlines() if line.strip())


def append_failed(game_id: str) -> None:
    with open(FAILED_FILE, "a") as f:
        f.write(game_id + "\n")


def remove_from_failed(game_id: str) -> None:
    if not FAILED_FILE.exists():
        return
    lines = FAILED_FILE.read_text().splitlines()
    FAILED_FILE.write_text("\n".join(l for l in lines if l.strip() != game_id) + "\n")


def run(seasons: list[str], resume: bool = False) -> None:
    game_rows = get_game_ids(seasons)

    if resume:
        failed = load_failed()
        if not failed:
            print("No failed games to resume.")
            return
        game_rows = [(gid, s) for gid, s in game_rows if gid in failed]
        print(f"Resuming {len(game_rows)} previously failed games.")

    conn = get_connection(PLAYERS_DB)
    setup_schema(conn)

    total = len(game_rows)
    newly_failed = []

    for i, (game_id, season) in enumerate(game_rows, 1):
        if is_cached(conn, game_id):
            print(f"[{i}/{total}] {game_id} ({season}) — CACHED")
            continue

        success = False
        for attempt in range(2):
            try:
                fetch_game_box_scores(conn, game_id, season)
                success = True
                if resume:
                    remove_from_failed(game_id)
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt == 0:
                    print(f"[{i}/{total}] {game_id} — network error, retrying in 5s: {e}")
                    time.sleep(5)
                else:
                    print(f"[{i}/{total}] {game_id} — FAILED after retry: {e}")
            except Exception as e:
                print(f"[{i}/{total}] {game_id} — FAILED (unexpected): {e}")
                break

        if success:
            print(f"[{i}/{total}] {game_id} ({season}) — FETCHED")
        else:
            append_failed(game_id)
            newly_failed.append(game_id)

        time.sleep(0.6)

    conn.close()

    if newly_failed:
        print(f"\nWARNING: {len(newly_failed)} games failed. See {FAILED_FILE}")
        print("Re-run with --resume to retry them.")
    else:
        print("\nAll player box scores fetched successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NBA player box scores.")
    parser.add_argument("--season", help="Fetch a single season (e.g. 2023-24). Defaults to all seasons.")
    parser.add_argument("--resume", action="store_true", help="Only retry games in failed_players.txt.")
    args = parser.parse_args()

    seasons_to_fetch = [args.season] if args.season else SEASONS
    if args.season and args.season not in SEASONS:
        raise ValueError(f"Unknown season '{args.season}'. Valid options: {SEASONS}")

    run(seasons_to_fetch, resume=args.resume)
