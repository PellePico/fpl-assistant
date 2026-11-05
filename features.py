"""Features for the machine-learning points model.

The model predicts a player's points in ONE fixture. Its inputs (features) are:

  * season-to-date stats of the player, turned into rates
    (per 90 minutes, or per gameweek), e.g. expected goals per 90
  * the fixture: difficulty (1-5) and home/away
  * the player's position (one-hot: is_gkp, is_def, is_mid, is_fwd)

The most important rule: the features at training time (built from past seasons'
gameweek-by-gameweek data) must be computed EXACTLY like the features at prediction
time (built from the live FPL API, which only gives season totals). So both paths
first produce the same small "season totals" table, and then call ONE shared
function, build_features(), that turns totals + fixtures into features.

    history (merged_gw.csv, gameweeks < g)  -> totals_from_history() -+
                                                                      +-> build_features()
    live API (bootstrap-static players)     -> totals_from_live()    -+
"""

from pathlib import Path

import pandas as pd
import requests

from fpl_data import POSITIONS, to_team_games
from points_model import per_90

# Season-to-date totals we use. All exist in the live API and (summed) in merged_gw.csv.
STATS = ["minutes", "starts", "total_points", "expected_goals", "expected_assists",
         "expected_goals_conceded", "bonus", "bps", "saves", "ict_index",
         "defensive_contribution"]

# The columns the model sees, in this exact order (training and prediction).
FEATURES = [
    # how much the player plays
    "minutes_per_gw", "starts_per_gw",
    # how many points they have scored
    "points_per_gw", "form",
    # underlying stats per 90 minutes (with the 270-minute floor from points_model)
    "xg90", "xa90", "xgc90", "bonus90", "bps90", "saves90", "ict90", "dc90",
    # the fixture
    "difficulty", "is_home",
    # position, one-hot
    "is_gkp", "is_def", "is_mid", "is_fwd",
]

FORM_GWS = 4          # FPL "form" = average points over roughly the last 30 days (~4 gameweeks)

RAW_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/{file}"
RAW_DIR = Path(__file__).parent / "data" / "raw"


# ---------------------------------------------------------------- shared part

def build_features(totals: pd.DataFrame, team_games: pd.DataFrame,
                   gws: list[int], finished_gws: int) -> pd.DataFrame:
    """One row per (player, upcoming fixture) with the FEATURES columns.

    totals: one row per player with columns id, team, position, form and STATS
            (season-to-date sums, only using gameweeks that are finished).
    team_games: from fpl_data.to_team_games (team, event, opponent, venue, difficulty).
    gws: the gameweeks to predict. finished_gws: how many gameweeks are played so far.
    """
    t = totals.copy()
    gws_so_far = max(finished_gws, 1)                  # avoid dividing by zero in gameweek 1

    t["minutes_per_gw"] = t["minutes"] / gws_so_far
    t["starts_per_gw"] = t["starts"] / gws_so_far
    t["points_per_gw"] = t["total_points"] / gws_so_far

    t["xg90"] = per_90(t["expected_goals"], t["minutes"])
    t["xa90"] = per_90(t["expected_assists"], t["minutes"])
    t["xgc90"] = per_90(t["expected_goals_conceded"], t["minutes"])
    t["bonus90"] = per_90(t["bonus"], t["minutes"])
    t["bps90"] = per_90(t["bps"], t["minutes"])
    t["saves90"] = per_90(t["saves"], t["minutes"])
    t["ict90"] = per_90(t["ict_index"], t["minutes"])
    t["dc90"] = per_90(t["defensive_contribution"], t["minutes"])

    for pos in ["GKP", "DEF", "MID", "FWD"]:
        t[f"is_{pos.lower()}"] = (t["position"] == pos).astype(int)

    # Every player x each of their team's games in those gameweeks
    # (a blank gameweek gives no row, a double gameweek gives two rows).
    games = team_games[team_games["event"].isin(gws)]
    rows = t.merge(games, on="team")
    rows["is_home"] = (rows["venue"] == "H").astype(int)

    return rows[["id", "team", "event", "opponent"] + FEATURES].reset_index(drop=True)


# ---------------------------------------------------------------- prediction time (live API)

def totals_from_live(players: pd.DataFrame) -> pd.DataFrame:
    """Season totals from the live bootstrap-static players table (fpl_data.load_season)."""
    t = pd.DataFrame({"id": players["id"], "team": players["team"],
                      "position": players["element_type"].map(POSITIONS)})
    for col in STATS + ["form"]:
        # Some API numbers arrive as text (e.g. ict_index "17.4"), so convert everything.
        t[col] = pd.to_numeric(players[col], errors="coerce").fillna(0)
    return t


def features_from_live(players: pd.DataFrame, team_games: pd.DataFrame,
                       gws: list[int], finished_gws: int) -> pd.DataFrame:
    return build_features(totals_from_live(players), team_games, gws, finished_gws)


# ---------------------------------------------------------------- training time (history)

def totals_from_history(merged: pd.DataFrame, g: int) -> pd.DataFrame:
    """Season totals as they were just before gameweek g, using ONLY gameweeks < g.

    merged: a season's merged_gw.csv with an extra 'team_id' column (see load_history).
    Returns one row per player who is listed in gameweek g (team = their club in g).
    """
    past = merged[merged["GW"] < g]
    totals = past.groupby("element")[STATS].sum()

    # Form: points per gameweek over the last FORM_GWS gameweeks
    # (the live API gives FPL's own "form", which is roughly the same thing).
    recent = past[past["GW"] >= g - FORM_GWS]
    form = recent.groupby("element")["total_points"].sum() / min(FORM_GWS, g - 1)

    # The players who are in the game in gameweek g, with their club and position then.
    now = merged[merged["GW"] == g].drop_duplicates("element")
    t = pd.DataFrame({"id": now["element"], "team": now["team_id"],
                      "position": now["position"].replace({"GK": "GKP"})})
    t = t.merge(totals, left_on="id", right_index=True, how="left")
    t["form"] = t["id"].map(form)
    return t.fillna(0)                  # no history yet (new signing) -> all zeros


def features_from_history(merged: pd.DataFrame, team_games: pd.DataFrame, g: int) -> pd.DataFrame:
    """Features for gameweek g + the actual points ('target') of each fixture."""
    x = build_features(totals_from_history(merged, g), team_games, gws=[g], finished_gws=g - 1)

    # Attach what really happened: match on player, gameweek and opponent
    # (the opponent tells the two games of a double gameweek apart).
    actual = merged[merged["GW"] == g][["element", "GW", "opponent", "total_points", "minutes", "xP"]]
    actual = actual.rename(columns={"element": "id", "GW": "event", "total_points": "target"})
    return x.merge(actual, on=["id", "event", "opponent"], how="inner")


# ---------------------------------------------------------------- downloading history

def _download(season: str, file: str) -> Path:
    """Download a file from the vaastav/Fantasy-Premier-League repo once, then reuse it."""
    path = RAW_DIR / season / Path(file).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(RAW_URL.format(season=season, file=file), timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def load_history(season: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    """Return (merged_gw, team_games, teams) for a past season, e.g. '2025-26'.

    merged_gw gets two helper columns: team_id (its 'team' column is a club name)
    and opponent (short name, like in team_games).
    """
    merged = pd.read_csv(_download(season, "gws/merged_gw.csv"))
    fixtures = pd.read_csv(_download(season, "fixtures.csv"))
    teams_csv = pd.read_csv(_download(season, "teams.csv"))

    teams = dict(zip(teams_csv["id"], teams_csv["short_name"]))
    merged["team_id"] = merged["team"].map(dict(zip(teams_csv["name"], teams_csv["id"])))
    merged["opponent"] = merged["opponent_team"].map(teams)
    if merged["team_id"].isna().any():
        raise ValueError(f"{season}: some club names in merged_gw.csv are not in teams.csv")

    return merged, to_team_games(fixtures, teams), teams
