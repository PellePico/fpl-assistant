"""Fetch live data from the official (public, no key) Fantasy Premier League API."""

import pandas as pd
import requests

BASE = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "fpl-assistant (student project)"}
SHIRT_URL = "https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{code}{gk}-110.webp"

POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
NUMERIC = ["expected_goals", "expected_assists", "expected_goals_conceded",
           "form", "points_per_game", "selected_by_percent", "ep_next"]


def _get(path: str) -> dict | list:
    response = requests.get(f"{BASE}/{path}", headers=HEADERS, timeout=20)
    response.raise_for_status()
    return response.json()


def load_season() -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Return (players, team_games, next_gameweek, finished_gameweeks)."""
    data = _get("bootstrap-static/")

    teams = {t["id"]: t["short_name"] for t in data["teams"]}
    players = pd.DataFrame(data["elements"])
    players[NUMERIC] = players[NUMERIC].apply(pd.to_numeric, errors="coerce")
    players["name"] = players["web_name"]
    players["position"] = players["element_type"].map(POSITIONS)
    players["club"] = players["team"].map(teams)
    players["price"] = players["now_cost"] / 10          # API stores £5.5m as 55
    players["shirt"] = [SHIRT_URL.format(code=c, gk="_1" if pos == 1 else "")
                        for c, pos in zip(players["team_code"], players["element_type"])]

    finished = sum(e["finished"] for e in data["events"])
    next_gw = next(e["id"] for e in data["events"] if not e["finished"])

    fixtures = pd.DataFrame(_get("fixtures/?future=1"))
    return players, to_team_games(fixtures, teams), next_gw, finished


def to_team_games(fixtures: pd.DataFrame, teams: dict[int, str]) -> pd.DataFrame:
    """One row per team per upcoming game: team, event, opponent, venue, difficulty.

    The API lists each match once (home vs away); we need it from both teams' view.
    """
    fixtures = fixtures.dropna(subset=["event"])          # postponed games have no gameweek
    home = pd.DataFrame({"team": fixtures["team_h"], "event": fixtures["event"],
                         "opponent": fixtures["team_a"].map(teams), "venue": "H",
                         "difficulty": fixtures["team_h_difficulty"]})
    away = pd.DataFrame({"team": fixtures["team_a"], "event": fixtures["event"],
                         "opponent": fixtures["team_h"].map(teams), "venue": "A",
                         "difficulty": fixtures["team_a_difficulty"]})
    games = pd.concat([home, away], ignore_index=True)
    games["event"] = games["event"].astype(int)
    return games.sort_values(["team", "event"]).reset_index(drop=True)


def load_my_team(entry_id: int, gameweek: int) -> tuple[list[int], float]:
    """Your 15 player ids and money in the bank (£m) after the given gameweek."""
    team = load_my_lineup(entry_id, gameweek)
    return team["squad"], team["bank"]


def load_my_lineup(entry_id: int, gameweek: int) -> dict:
    """Your squad as you set it: squad, xi (positions 1-11), captain, vice, bank."""
    data = _get(f"entry/{entry_id}/event/{gameweek}/picks/")
    picks = sorted(data["picks"], key=lambda p: p["position"])
    return {
        "squad": [p["element"] for p in picks],
        "xi": [p["element"] for p in picks if p["position"] <= 11],
        "captain": next(p["element"] for p in picks if p["is_captain"]),
        "vice": next(p["element"] for p in picks if p["is_vice_captain"]),
        "bank": data["entry_history"]["bank"] / 10,
    }


def load_deadline(gameweek: int) -> str:
    """The deadline of a gameweek, e.g. 'Sat 7 Nov, 12:30' (UK time from the API, shown in CPH time)."""
    events = _get("bootstrap-static/")["events"]
    deadline = pd.Timestamp(next(e["deadline_time"] for e in events if e["id"] == gameweek))
    return deadline.tz_convert("Europe/Copenhagen").strftime("%a %-d %b, %H:%M")
