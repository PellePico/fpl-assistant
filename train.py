"""Train the machine-learning points model on last season (2025-26).

    python train.py

For every gameweek g from FIRST_GW to 38 we build the features using only
gameweeks before g, and the target = the points the player really scored in g.
Stacking all these gameweeks gives one big table (one row per player per fixture).

The model is a gradient-boosted tree ensemble (many small decision trees, each one
correcting the errors of the previous ones). It predicts the AVERAGE points
(squared-error loss), which is what the optimizer needs: expected points.

Note on injuries: the historical data has no "chance of playing" flags, so the model
learns from all rows as they are (injured players are simply rows with 0 points).
At prediction time we multiply by the player's availability instead (see ml_model.py).
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import FEATURES, features_from_history, load_history

TRAIN_SEASON = "2025-26"
FIRST_GW = 4               # from gameweek 4 there are at least 3 gameweeks of history
LAST_GW = 38
MODEL_PATH = Path(__file__).parent / "models" / "points_model.joblib"


def training_table(season: str, first_gw: int, last_gw: int) -> pd.DataFrame:
    merged, team_games, _ = load_history(season)
    tables = [features_from_history(merged, team_games, g) for g in range(first_gw, last_gw + 1)]
    return pd.concat(tables, ignore_index=True)


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",      # predict the mean -> expected points
        learning_rate=0.05,
        max_iter=300,              # number of trees
        max_leaf_nodes=15,         # small trees -> less overfitting
        min_samples_leaf=50,       # every leaf averages at least 50 player-fixtures
        l2_regularization=1.0,
        random_state=0,
    )


def main():
    data = training_table(TRAIN_SEASON, FIRST_GW, LAST_GW)
    print(f"Training rows: {len(data)} (player-fixtures, {TRAIN_SEASON} GW{FIRST_GW}-{LAST_GW})")

    model = make_model()
    model.fit(data[FEATURES], data["target"])

    MODEL_PATH.parent.mkdir(exist_ok=True)
    # Save the feature list with the model, so prediction can check it uses the same columns.
    joblib.dump({"model": model, "features": FEATURES, "trained_on": TRAIN_SEASON}, MODEL_PATH)
    print(f"Saved {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
