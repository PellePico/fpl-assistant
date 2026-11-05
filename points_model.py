"""Expected points (xP) per player per gameweek.

FPL points come from a few sources. We estimate each one separately, so every
prediction can be explained as a sum of parts:

    xP = appearance + goals + assists + clean sheet + saves + bonus - goals conceded

Each part = (how often it happens per 90 minutes) x (points it's worth)
            x (share of the match they play) x (fixture difficulty adjustment)
"""

import numpy as np
import pandas as pd

GOAL_POINTS = {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}
CLEAN_SHEET_POINTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_POINTS = 3

# Fixture difficulty (1 = easiest, 5 = hardest) -> how much easier/harder to score.
DIFFICULTY_MULTIPLIER = {1: 1.3, 2: 1.15, 3: 1.0, 4: 0.85, 5: 0.7}

# Per-90 stats from a few minutes are noise (1 goal in 20 min = 4.5 goals/90).
# Divide by at least this many minutes, which pulls small samples towards zero.
MIN_MINUTES = 270

PARTS = ["appearance", "goals", "assists", "clean_sheet", "saves", "bonus", "conceded"]


def per_90(total: pd.Series, minutes: pd.Series) -> pd.Series:
    return total * 90 / np.maximum(minutes, MIN_MINUTES)


def add_expected_points(players: pd.DataFrame, team_games: pd.DataFrame,
                        next_gw: int, finished_gws: int, horizon: int = 6) -> pd.DataFrame:
    """Add xP columns to the players table.

    gw10, gw11, ...   xP per gameweek (a blank gameweek is 0, a double is two games)
    xp_next           xP for the next gameweek
    xp3               xP for the next 3 gameweeks (what the optimizer plans with)
    next_<part>       next gameweek's xP split into its parts, e.g. next_goals
    """
    p = players.copy()

    # Expected minutes = average minutes per gameweek so far x chance of being available.
    chance = p["chance_of_playing_next_round"]
    available = np.where(chance.isna(), (p["status"] == "a").astype(float), chance / 100)
    p["exp_minutes"] = np.minimum(p["minutes"] / max(finished_gws, 1), 90) * available

    p["xg90"] = per_90(p["expected_goals"], p["minutes"])
    p["xa90"] = per_90(p["expected_assists"], p["minutes"])
    p["xgc90"] = per_90(p["expected_goals_conceded"], p["minutes"])
    p["bonus90"] = per_90(p["bonus"], p["minutes"])
    p["saves90"] = per_90(p["saves"], p["minutes"])

    # Every player x each of their upcoming games.
    gws = list(range(next_gw, next_gw + horizon))
    rows = p.merge(team_games[team_games["event"].isin(gws)], on="team")
    rows[PARTS] = fixture_parts(rows)
    rows["xp"] = rows[PARTS].sum(axis=1)

    per_gw = rows.pivot_table(index="id", columns="event", values="xp", aggfunc="sum")
    per_gw = per_gw.reindex(columns=gws).fillna(0)        # blank gameweek = 0 points
    per_gw.columns = [f"gw{gw}" for gw in gws]
    p = p.merge(per_gw, left_on="id", right_index=True, how="left")
    p[per_gw.columns] = p[per_gw.columns].fillna(0)

    p["xp_next"] = p[f"gw{next_gw}"]
    p["xp3"] = p[[f"gw{gw}" for gw in gws[:3]]].sum(axis=1)

    nxt = rows[rows["event"] == next_gw].groupby("id")[PARTS].sum().add_prefix("next_")
    p = p.merge(nxt, left_on="id", right_index=True, how="left")
    p[nxt.columns] = p[nxt.columns].fillna(0)
    return p


def fixture_parts(r: pd.DataFrame) -> pd.DataFrame:
    """Expected points from each source, for each (player, game) row."""
    mult = r["difficulty"].map(DIFFICULTY_MULTIPLIER)
    share = r["exp_minutes"] / 90                          # fraction of a match played
    plays_60 = np.minimum(r["exp_minutes"] / 60, 1)        # roughly: chance of 60+ minutes
    goals_against = r["xgc90"] / mult                      # harder opponent -> concede more

    return pd.DataFrame({
        "appearance": 2 * plays_60,                                         # 2 pts for 60+ min
        "goals": r["xg90"] * share * mult * r["position"].map(GOAL_POINTS),
        "assists": r["xa90"] * share * mult * ASSIST_POINTS,
        # Poisson: P(0 goals conceded) = e^(-expected goals conceded)
        "clean_sheet": np.exp(-goals_against) * plays_60 * r["position"].map(CLEAN_SHEET_POINTS),
        "saves": np.where(r["position"] == "GKP", r["saves90"] / 3 * share, 0),  # 1 pt per 3 saves
        "bonus": r["bonus90"] * share,
        "conceded": np.where(r["position"].isin(["GKP", "DEF"]), -goals_against / 2 * share, 0),
    }, index=r.index)
