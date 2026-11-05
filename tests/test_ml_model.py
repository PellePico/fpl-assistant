import pandas as pd
import pytest

from features import FEATURES, features_from_history, features_from_live
from fpl_data import to_team_games
from ml_model import predict_points

TEAMS = {1: "AAA", 2: "BBB", 3: "CCC", 4: "DDD"}


class FakeModel:
    """Stands in for the trained model, so the tests run offline and fast.
    Predicts 1 point + 1 point per 30 minutes played per gameweek."""
    def predict(self, X):
        assert list(X.columns) == FEATURES
        return 1 + X["minutes_per_gw"].to_numpy() / 30


def games(*rows):
    cols = ["event", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]
    return to_team_games(pd.DataFrame(rows, columns=cols), TEAMS)


def live_player(id_, team, **overrides):
    """A row like the bootstrap-static players table (note: ict_index is text in the API)."""
    base = {"id": id_, "team": team, "element_type": 3, "status": "a",
            "chance_of_playing_next_round": None, "minutes": 270, "starts": 3,
            "total_points": 15, "expected_goals": 1.2, "expected_assists": 0.6,
            "expected_goals_conceded": 3.0, "bonus": 2, "bps": 60, "saves": 0,
            "ict_index": "20.5", "defensive_contribution": 12, "form": 5.0}
    return base | overrides


def predict(players, fixtures, horizon=2):
    return predict_points(pd.DataFrame(players), fixtures, next_gw=10, finished_gws=9,
                          horizon=horizon, model=FakeModel())


def test_output_has_one_row_per_player_and_one_column_per_gameweek():
    result = predict([live_player(1, 1), live_player(2, 2)],
                     games((10, 1, 2, 3, 3), (11, 2, 1, 3, 3)), horizon=3)
    assert list(result.columns) == ["gw10", "gw11", "gw12"]
    assert sorted(result.index) == [1, 2]


def test_blank_gameweek_is_zero_and_double_is_two_games():
    result = predict([live_player(1, 1), live_player(2, 2), live_player(3, 3)],
                     games((10, 1, 3, 3, 3), (10, 1, 4, 3, 3)))   # team 1 twice, team 2 blank
    assert result.at[2, "gw10"] == 0
    assert result.at[1, "gw10"] == pytest.approx(2 * result.at[3, "gw10"])
    assert (result["gw11"] == 0).all()                            # nobody plays GW11


def test_injured_player_scores_zero_and_doubtful_is_scaled():
    result = predict([live_player(1, 1, status="i", chance_of_playing_next_round=0),
                      live_player(2, 1, status="d", chance_of_playing_next_round=50),
                      live_player(3, 1)],
                     games((10, 1, 2, 3, 3)))
    assert result.at[1, "gw10"] == 0
    assert result.at[2, "gw10"] == pytest.approx(result.at[3, "gw10"] / 2)


def test_training_and_live_features_are_identical():
    """The same player, seen through history (gameweek rows) and through the live API
    (season totals), must give exactly the same feature row."""
    gw_rows = pd.DataFrame({
        "element": [1, 1, 1], "GW": [1, 2, 3], "team_id": [1, 1, 1], "position": ["GK"] * 3,
        "opponent": ["BBB", "CCC", "DDD"], "minutes": [90, 90, 90], "starts": [1, 1, 1],
        "total_points": [6, 2, 3], "expected_goals": [0.0, 0.1, 0.0],
        "expected_assists": [0.0, 0.0, 0.05], "expected_goals_conceded": [0.5, 1.5, 1.0],
        "bonus": [2, 0, 0], "bps": [30, 10, 15], "saves": [4, 2, 3],
        "ict_index": [3.0, 1.0, 2.0], "defensive_contribution": [0, 1, 0], "xP": [3.0, 3.0, 3.0],
    })
    fixtures = games((4, 1, 2, 2, 4))

    # GW4 itself must be in the data to have a target; add it (its stats must NOT leak in).
    gw4 = gw_rows.iloc[[0]].assign(GW=4, opponent="BBB", total_points=99, minutes=90)
    history = features_from_history(pd.concat([gw_rows, gw4]), fixtures, g=4)

    live = features_from_live(pd.DataFrame([{
        "id": 1, "team": 1, "element_type": 1, "minutes": 270, "starts": 3, "total_points": 11,
        "expected_goals": 0.1, "expected_assists": 0.05, "expected_goals_conceded": 3.0,
        "bonus": 2, "bps": 55, "saves": 9, "ict_index": "6.0", "defensive_contribution": 1,
        "form": 11 / 3,   # FPL's form ~ points per gameweek lately (3 gameweeks played)
    }]), fixtures, gws=[4], finished_gws=3)

    assert list(live.columns) == list(history.columns[:len(live.columns)])
    assert history.at[0, "target"] == 99
    pd.testing.assert_frame_equal(history[FEATURES], live[FEATURES], check_dtype=False)


def test_saved_model_loads_and_predicts():
    from ml_model import MODEL_PATH, load_model
    if not MODEL_PATH.exists():
        pytest.skip("run train.py first")
    rows = features_from_live(pd.DataFrame([live_player(1, 1)]), games((10, 1, 2, 3, 3)),
                              gws=[10], finished_gws=9)
    prediction = load_model().predict(rows[FEATURES])
    assert 0 < prediction[0] < 15
