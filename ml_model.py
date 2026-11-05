"""Expected points per player per gameweek from the trained ML model.

The app calls predict_points(). The model itself is trained by train.py and saved in
models/points_model.joblib; it is loaded the first time it is needed.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import FEATURES, features_from_live

MODEL_PATH = Path(__file__).parent / "models" / "points_model.joblib"
_saved = None              # the loaded model, kept so we only read the file once


def load_model():
    global _saved
    if _saved is None:
        _saved = joblib.load(MODEL_PATH)
        if _saved["features"] != FEATURES:
            raise ValueError("Model was trained with other features - run train.py again")
    return _saved["model"]


def availability(players: pd.DataFrame) -> pd.Series:
    """Chance the player is available: chance_of_playing / 100.

    If FPL gives no chance, the player is fit when status is 'a' (available), else
    (injured, suspended, left the club, ...) we assume they won't play.
    Same rule as points_model.py.
    """
    chance = players["chance_of_playing_next_round"]
    fit = (players["status"] == "a").astype(float)
    return pd.Series(np.where(chance.isna(), fit, chance / 100), index=players["id"])


def predict_points(players: pd.DataFrame, team_games: pd.DataFrame, next_gw: int,
                   finished_gws: int, horizon: int = 6, model=None) -> pd.DataFrame:
    """Expected points, indexed by player id, columns gw{next_gw} ... gw{next_gw+horizon-1}.

    A gameweek's value is the sum over the player's fixtures in it
    (blank gameweek = 0, double gameweek = two fixtures added up).
    `model` can be passed in (used by the tests); normally the saved model is used.
    """
    model = model if model is not None else load_model()
    gws = list(range(next_gw, next_gw + horizon))

    # 1. One row per player per fixture, with the same features as in training.
    rows = features_from_live(players, team_games, gws, finished_gws)

    # 2. Model prediction for each fixture, times the chance the player is available.
    #    (The model never saw injury flags, so availability is applied here.)
    rows["xp"] = model.predict(rows[FEATURES]) if len(rows) else []
    rows["xp"] = rows["xp"] * rows["id"].map(availability(players))

    # 3. Sum per player per gameweek; gameweeks without a fixture become 0.
    per_gw = rows.pivot_table(index="id", columns="event", values="xp", aggfunc="sum")
    per_gw = per_gw.reindex(index=players["id"], columns=gws).fillna(0)
    per_gw.columns = [f"gw{gw}" for gw in gws]
    per_gw.columns.name = None
    return per_gw
