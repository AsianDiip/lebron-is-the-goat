"""
fetch_pbp.py — Fetch play-by-play data for all historical games.

Fetches PlayByPlayV3 for each game_id in games.db → pbp.db / play_by_play.

Run:
    python data/fetch_pbp.py [--season 2023-24] [--resume]

--resume: only attempt game_ids listed in data/raw/failed_games.txt
"""

import argparse
import random
import re
import sqlite3
import time
from pathlib import Path

import requests
from nba_api.stats.endpoints import PlayByPlayV3

SEASONS = [
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
]
GAMES_DB = Path(__file__).parent / "raw" / "games.db"
PBP_DB = Path(__file__).parent / "raw" / "pbp.db"
PLAYERS_DB = Path(__file__).parent / "raw" / "players.db"
FAILED_GAMES_FILE = Path(__file__).parent / "raw" / "failed_games.txt"

CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")


def get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def setup_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS play_by_play (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id       TEXT    NOT NULL,
            season        TEXT    NOT NULL,
            action_number INTEGER NOT NULL,
            action_id     INTEGER,
            period        INTEGER NOT NULL,
            clock_str     TEXT    NOT NULL,
            clock_seconds INTEGER NOT NULL,
            team_id       INTEGER,
            team_tricode  TEXT,
            person_id     INTEGER,
            player_name   TEXT,
            action_type   TEXT    NOT NULL,
            sub_type      TEXT,
            description   TEXT,
            score_home    INTEGER,
            score_away    INTEGER,
            is_field_goal INTEGER,
            shot_result   TEXT,
            UNIQUE(game_id, action_number)
        );
        CREATE INDEX IF NOT EXISTS idx_pbp_game_id     ON play_by_play(game_id);
        CREATE INDEX IF NOT EXISTS idx_pbp_season      ON play_by_play(season);
        CREATE INDEX IF NOT EXISTS idx_pbp_period      ON play_by_play(period);
        CREATE INDEX IF NOT EXISTS idx_pbp_action_type ON play_by_play(action_type);
    """)
    conn.commit()


def parse_clock(clock_str: str, game_id: str, action_number: int) -> int:
    """Parse ISO 8601 clock string to seconds remaining in period."""
    if not clock_str:
        return -1
    m = CLOCK_RE.match(clock_str)
    if not m:
        print(f"  WARNING: malformed clock '{clock_str}' in game {game_id} action {action_number}")
        return -1
    minutes = int(m.group(1))
    seconds = float(m.group(2))
    return int(minutes * 60 + seconds)


def get_game_ids(seasons: list[str]) -> list[tuple[str, str]]:
    """Return [(game_id, season), ...] from games.db."""
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
            f"No game_ids found in games.db for seasons {seasons}. Run fetch_games.py first."
        )
    return rows


def is_cached(conn: sqlite3.Connection, game_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM play_by_play WHERE game_id = ? LIMIT 1", (game_id,)
    ).fetchone() is not None


def fetch_game_pbp(pbp_conn: sqlite3.Connection, game_id: str, season: str) -> None:
    df = PlayByPlayV3(game_id=game_id).get_data_frames()[0]

    rows = []
    for _, row in df.iterrows():
        clock_str = str(row.get("clock", "") or "")
        action_num = int(row.get("actionNumber", 0))
        clock_sec = parse_clock(clock_str, game_id, action_num)

        # score fields only present on scoring plays
        score_home = row.get("scoreHome")
        score_away = row.get("scoreAway")
        score_home = int(score_home) if score_home == score_home and score_home is not None and str(score_home) != "" else None
        score_away = int(score_away) if score_away == score_away and score_away is not None and str(score_away) != "" else None

        team_id = row.get("teamId")
        team_id = int(team_id) if team_id == team_id and team_id is not None else None

        person_id = row.get("personId")
        person_id = int(person_id) if person_id == person_id and person_id is not None else None

        action_id = row.get("actionId")
        action_id = int(action_id) if action_id == action_id and action_id is not None else None

        rows.append((
            game_id,
            season,
            action_num,
            action_id,
            int(row.get("period", 0)),
            clock_str,
            clock_sec,
            team_id,
            row.get("teamTricode") or None,
            person_id,
            row.get("playerNameI") or None,
            str(row.get("actionType", "") or ""),
            row.get("subType") or None,
            row.get("description") or None,
            score_home,
            score_away,
            int(row.get("isFieldGoal", 0) or 0),
            row.get("shotResult") or None,
        ))

    pbp_conn.execute("BEGIN")
    pbp_conn.executemany(
        """INSERT OR IGNORE INTO play_by_play
           (game_id, season, action_number, action_id, period, clock_str, clock_seconds,
            team_id, team_tricode, person_id, player_name, action_type, sub_type,
            description, score_home, score_away, is_field_goal, shot_result)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    pbp_conn.execute("COMMIT")


def load_failed_games() -> set[str]:
    if not FAILED_GAMES_FILE.exists():
        return set()
    return set(line.strip() for line in FAILED_GAMES_FILE.read_text().splitlines() if line.strip())


def append_failed_game(game_id: str) -> None:
    with open(FAILED_GAMES_FILE, "a") as f:
        f.write(game_id + "\n")


def remove_from_failed(game_id: str) -> None:
    if not FAILED_GAMES_FILE.exists():
        return
    lines = FAILED_GAMES_FILE.read_text().splitlines()
    FAILED_GAMES_FILE.write_text("\n".join(l for l in lines if l.strip() != game_id) + "\n")


def run(seasons: list[str], resume: bool = False) -> None:
    game_rows = get_game_ids(seasons)

    if resume:
        failed = load_failed_games()
        if not failed:
            print("No failed games to resume.")
            return
        game_rows = [(gid, s) for gid, s in game_rows if gid in failed]
        print(f"Resuming {len(game_rows)} previously failed games.")

    pbp_conn = get_connection(PBP_DB)
    setup_schema(pbp_conn)

    total = len(game_rows)
    newly_failed = []

    for i, (game_id, season) in enumerate(game_rows, 1):
        if is_cached(pbp_conn, game_id):
            print(f"[{i}/{total}] {game_id} ({season}) — CACHED")
            continue

        success = False
        for attempt in range(2):
            try:
                fetch_game_pbp(pbp_conn, game_id, season)
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
            append_failed_game(game_id)
            newly_failed.append(game_id)

        time.sleep(0.6)

    pbp_conn.close()

    if newly_failed:
        print(f"\nWARNING: {len(newly_failed)} games failed. See {FAILED_GAMES_FILE}")
        print("Re-run with --resume to retry them.")
    else:
        print(f"\nAll games fetched successfully.")

    validate_cross_table_consistency()


def validate_cross_table_consistency() -> None:
    """Assert game_id coverage is consistent across all three databases."""
    if not GAMES_DB.exists():
        print("Validation skipped: games.db not found.")
        return
    if not PBP_DB.exists():
        print("Validation skipped: pbp.db not found.")
        return

    conn_games = sqlite3.connect(GAMES_DB)
    games_ids = set(
        r[0] for r in conn_games.execute("SELECT DISTINCT game_id FROM game_logs")
    )
    conn_games.close()

    conn_pbp = sqlite3.connect(PBP_DB)
    pbp_ids = set(
        r[0] for r in conn_pbp.execute("SELECT DISTINCT game_id FROM play_by_play")
    )

    missing_pbp = games_ids - pbp_ids
    if missing_pbp:
        raise ValueError(
            f"{len(missing_pbp)} games in game_logs have no play-by-play data.\n"
            f"First 20 missing: {sorted(missing_pbp)[:20]}"
        )

    orphan_pbp = pbp_ids - games_ids
    if orphan_pbp:
        raise ValueError(
            f"{len(orphan_pbp)} game_ids in play_by_play are not in game_logs: "
            f"{sorted(orphan_pbp)[:10]}"
        )

    # Per-game sanity check on a sample
    sample_ids = random.sample(sorted(pbp_ids), min(50, len(pbp_ids)))
    for gid in sample_ids:
        pbp_count = conn_pbp.execute(
            "SELECT COUNT(*) FROM play_by_play WHERE game_id=?", (gid,)
        ).fetchone()[0]
        if pbp_count < 100:
            raise ValueError(
                f"game_id {gid} has only {pbp_count} play-by-play rows — "
                "likely a corrupted or empty fetch. Re-run fetch_pbp.py."
            )
    conn_pbp.close()

    # Check player coverage if players.db exists
    player_coverage = "N/A (players.db not yet fetched)"
    player_pct = None
    if PLAYERS_DB.exists():
        conn_players = sqlite3.connect(PLAYERS_DB)
        player_ids = set(
            r[0] for r in conn_players.execute("SELECT DISTINCT game_id FROM player_box_scores")
        )
        conn_players.close()

        missing_players = games_ids - player_ids
        if missing_players:
            player_coverage = f"{len(games_ids) - len(missing_players)}/{len(games_ids)} ({100*(1 - len(missing_players)/len(games_ids)):.1f}%)"
        else:
            player_pct = 100.0
            player_coverage = f"{len(games_ids)}/{len(games_ids)} (100.0%)"

    print("\nValidation PASSED.")
    print(f"  Games in registry:      {len(games_ids):,}")
    print(f"  Games with PBP:         {len(pbp_ids):,} ({100*len(pbp_ids)/len(games_ids):.1f}%)")
    print(f"  Games with box scores:  {player_coverage}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NBA play-by-play data.")
    parser.add_argument("--season", help="Fetch a single season (e.g. 2023-24). Defaults to all seasons.")
    parser.add_argument("--resume", action="store_true", help="Only retry games in failed_games.txt.")
    args = parser.parse_args()

    seasons_to_fetch = [args.season] if args.season else SEASONS
    if args.season and args.season not in SEASONS:
        raise ValueError(f"Unknown season '{args.season}'. Valid options: {SEASONS}")

    run(seasons_to_fetch, resume=args.resume)
