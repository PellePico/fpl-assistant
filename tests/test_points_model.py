import pandas as pd
import pytest

from fpl_data import to_team_games
from points_model import PARTS, add_expected_points

TEAMS = {1: "AAA", 2: "BBB", 3: "CCC", 4: "DDD"}


def player(id_, team, minutes=810, **overrides):
    base = {
        "id": id_, "team": team, "position": "MID", "minutes": minutes, "status": "a",
        "chance_of_playing_next_round": None, "expected_goals": 3.0, "expected_assists": 2.0,
        "expected_goals_conceded": 9.0, "bonus": 5, "saves": 0,
    }
    return base | overrides


def games(*rows):
    cols = ["event", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]
    return to_team_games(pd.DataFrame(rows, columns=cols), TEAMS)


def xp(players, fixtures):
    return add_expected_points(pd.DataFrame(players), fixtures, next_gw=10, finished_gws=9,
                               horizon=1).set_index("id")


def test_easier_fixture_gives_more_points():
    result = xp([player(1, team=1), player(2, team=2)],
                games((10, 1, 3, 2, 4), (10, 2, 4, 5, 2)))   # team 1 easy (2), team 2 hard (5)
    assert result.at[1, "xp_next"] > result.at[2, "xp_next"]


def test_injured_player_scores_zero():
    result = xp([player(1, team=1, status="i", chance_of_playing_next_round=0)],
                games((10, 1, 2, 3, 3)))
    assert result.at[1, "xp_next"] == 0


def test_blank_gameweek_scores_zero_and_double_scores_more():
    result = xp([player(1, team=1), player(2, team=2), player(3, team=3)],
                games((10, 1, 3, 3, 3), (10, 1, 4, 3, 3)))    # team 1 plays twice, team 2 never
    assert result.at[2, "xp_next"] == 0
    assert result.at[1, "xp_next"] > result.at[3, "xp_next"] > 0


def test_parts_add_up_to_total():
    result = xp([player(1, team=1, position="DEF")], games((10, 1, 2, 3, 3)))
    total = sum(result.at[1, f"next_{part}"] for part in PARTS)
    assert total == pytest.approx(result.at[1, "xp_next"])
