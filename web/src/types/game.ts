export interface ProbPoint {
  seconds_remaining: number;
  home_win_prob: number;
  action_number: number;
}

export interface PlayEvent {
  action_number: number;
  period: number;
  clock_seconds: number;
  team_abbrev: string;
  description: string;
  home_score: number;
  away_score: number;
  home_win_prob: number;
}

export interface TeamBoxScore {
  fgm: number;
  fga: number;
  fg_pct: number;
  two_pm: number;
  two_pa: number;
  two_pt_pct: number;
  three_pm: number;
  three_pa: number;
  three_pt_pct: number;
  ftm: number;
  fta: number;
  ft_pct: number;
  fouls: number;
  turnovers: number;
  timeouts_remaining: number;
}

export interface BoxScoreData {
  home: TeamBoxScore;
  away: TeamBoxScore;
}

export interface GameState {
  game_id: string;
  period: number;
  clock_seconds: number;
  home_score: number;
  away_score: number;
  home_win_prob: number;
  events_processed: number;
  home_team_id: number;
  away_team_id: number;
  home_team_abbrev: string;
  away_team_abbrev: string;
  pre_game_prob: number;
  prob_history: ProbPoint[];
  play_log: PlayEvent[];
  box_score: BoxScoreData;
}

export interface PregameBreakdownData {
  game_id: string;
  home_team_id: number;
  away_team_id: number;
  pre_game_prob: number;
  features: Record<string, number>;
  labels: Record<string, string>;
}

export interface GameListItem {
  game_id: string;
  game_date: string;
  season: string;
  away_team: string;
  home_team: string;
  away_pts: number;
  home_pts: number;
  home_wl: string;
}

export interface GameListResponse {
  seasons: string[];
  games: GameListItem[];
}

export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "error";
