import type { ProbPoint, PlayEvent } from "@/types/game";

export interface MomentumSwing {
  index: number;
  delta: number;
  description: string;
  seconds_remaining: number;
  home_win_prob: number;
  home_score: number;
  away_score: number;
  period: number;
}

/**
 * Find the top N largest single-event probability swings.
 */
export function topSwings(
  probHistory: ProbPoint[],
  playLog: PlayEvent[],
  n: number = 5,
): MomentumSwing[] {
  if (probHistory.length < 2) return [];

  const swings: MomentumSwing[] = [];
  for (let i = 1; i < probHistory.length; i++) {
    const delta = Math.abs(probHistory[i].home_win_prob - probHistory[i - 1].home_win_prob);
    const play = playLog[i];
    if (play) {
      swings.push({
        index: i,
        delta,
        description: play.description,
        seconds_remaining: probHistory[i].seconds_remaining,
        home_win_prob: probHistory[i].home_win_prob,
        home_score: play.home_score,
        away_score: play.away_score,
        period: play.period,
      });
    }
  }

  swings.sort((a, b) => b.delta - a.delta);
  return swings.slice(0, n);
}

/**
 * Count lead changes from play log.
 */
export function countLeadChanges(playLog: PlayEvent[]): number {
  let changes = 0;
  let prevLeader: "home" | "away" | "tie" = "tie";

  for (const play of playLog) {
    const diff = play.home_score - play.away_score;
    const leader = diff > 0 ? "home" : diff < 0 ? "away" : "tie";
    if (leader !== "tie" && prevLeader !== "tie" && leader !== prevLeader) {
      changes++;
    }
    if (leader !== "tie") prevLeader = leader;
  }

  return changes;
}

/**
 * Find the largest scoring run (consecutive points by one team).
 */
export function largestRun(playLog: PlayEvent[]): { team: string; points: number } | null {
  if (playLog.length < 2) return null;

  let best = { team: "", points: 0 };
  let currentTeam = "";
  let currentRun = 0;

  for (let i = 1; i < playLog.length; i++) {
    const scoreDelta =
      (playLog[i].home_score + playLog[i].away_score) -
      (playLog[i - 1].home_score + playLog[i - 1].away_score);

    if (scoreDelta <= 0) continue;

    const homeDelta = playLog[i].home_score - playLog[i - 1].home_score;
    const scorer = homeDelta > 0 ? "home" : "away";

    if (scorer === currentTeam) {
      currentRun += scoreDelta;
    } else {
      currentTeam = scorer;
      currentRun = scoreDelta;
    }

    if (currentRun > best.points) {
      best = { team: currentTeam, points: currentRun };
    }
  }

  return best.points > 0 ? best : null;
}
