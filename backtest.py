"""Backtest: how well do different methods predict next gameweek's points?

    python backtest.py      (run train.py first)

Season 2026-27 (the model was trained on 2025-26 only, so this is unseen data).
For each gameweek g in TEST_GWS we predict g using only gameweeks before g,
then compare with the real points. We score per player per gameweek (a double
gameweek's two fixtures are added up), over every player listed in that gameweek.

Methods
  ml_model      the gradient-boosting model (ml_model.py)
  formula       the hand-made formula (points_model.py)
  fpl_xp        FPL's own expected points (xP). CAVEAT: scraped after the gameweek,
                so it may contain information from after kick-off (e.g. who started)
  last5_mean    average points over the previous 5 gameweeks
  zero          always predict 0 (most listed players don't play)

Nobody gets injury news here: the history has no "chance of playing" flags, so the
model and the formula treat every listed player as available. FPL's xP does know.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURES, features_from_history, load_history, totals_from_history
from ml_model import load_model
from points_model import add_expected_points

TEST_SEASON = "2026-27"
TEST_GWS = [3, 4, 5, 6]
RESULTS_PATH = Path(__file__).parent / "models" / "backtest_results.json"


def predictions_for_gw(merged: pd.DataFrame, team_games: pd.DataFrame, g: int) -> pd.DataFrame:
    """One row per player listed in gameweek g: actual points + every method's prediction."""
    this_gw = merged[merged["GW"] == g]
    actual = this_gw.groupby("element").agg(actual=("total_points", "sum"),
                                            minutes=("minutes", "sum"),
                                            fpl_xp=("xP", "first"))   # xP is per gameweek
    actual.index.name = "id"

    # (a) ML model: predict each fixture, add up per player.
    x = features_from_history(merged, team_games, g)
    x["pred"] = load_model().predict(x[FEATURES])
    actual["ml_model"] = x.groupby("id")["pred"].sum()

    # (b) Hand-made formula, fed with the same season-to-date totals.
    totals = totals_from_history(merged, g)
    totals["status"] = "a"                          # no injury info in the history
    totals["chance_of_playing_next_round"] = np.nan
    formula = add_expected_points(totals, team_games, next_gw=g, finished_gws=g - 1, horizon=1)
    actual["formula"] = formula.set_index("id")[f"gw{g}"]

    # (d) Average points per gameweek over the last 5 gameweeks the player was listed.
    past = merged[(merged["GW"] < g) & (merged["GW"] >= g - 5)]
    per_gw = past.groupby(["element", "GW"])["total_points"].sum()
    actual["last5_mean"] = per_gw.groupby("element").mean()

    # (e) Predict zero for everyone.
    actual["zero"] = 0.0

    actual["gw"] = g
    return actual.fillna({"ml_model": 0, "formula": 0, "last5_mean": 0})


def score(df: pd.DataFrame, method: str) -> dict:
    error = df[method] - df["actual"]
    played = df["minutes"] > 0
    return {"method": method,
            "mae": round(float(error.abs().mean()), 3),
            "rmse": round(float(np.sqrt((error ** 2).mean())), 3),
            "mae_played": round(float(error[played].abs().mean()), 3),
            "n": int(len(df))}


def main():
    merged, team_games, _ = load_history(TEST_SEASON)
    df = pd.concat([predictions_for_gw(merged, team_games, g) for g in TEST_GWS])

    methods = ["ml_model", "formula", "fpl_xp", "last5_mean", "zero"]
    results = [score(df, m) for m in methods]

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print(f"Backtest {TEST_SEASON} GW{TEST_GWS[0]}-{TEST_GWS[-1]}, one gameweek ahead "
          f"({(df['minutes'] > 0).sum()} of {len(df)} player-gameweeks played)\n")
    print(pd.DataFrame(results).set_index("method").to_string())
    print("\nfpl_xp may contain post-kick-off information (lookahead) - not a fair comparison.")


if __name__ == "__main__":
    main()
