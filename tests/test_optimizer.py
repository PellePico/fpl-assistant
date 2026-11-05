"""Tests run on made-up players, so they don't need the internet (works in CI)."""

import random

import pandas as pd

from optimizer import MAX_PER_CLUB, SQUAD_SIZE, XI_MIN, pick_lineup, pick_squad


def fake_players(n_per_position: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for pos in SQUAD_SIZE:
        for k in range(n_per_position):
            xp = rng.uniform(0, 6)
            rows.append({"id": len(rows) + 1, "position": pos, "club": f"C{k % 8}",
                         "price": round(rng.uniform(4.0, 13.0), 1), "xp_next": xp, "xp3": 3 * xp,
                         "can_select": True})
    return pd.DataFrame(rows)


def test_squad_follows_fpl_rules():
    players = fake_players()
    squad = players.set_index("id").loc[pick_squad(players, budget=100)["squad"]]

    assert len(squad) == 15
    assert squad["position"].value_counts().to_dict() == SQUAD_SIZE
    assert squad["price"].sum() <= 100
    assert squad["club"].value_counts().max() <= MAX_PER_CLUB


def test_lineup_is_a_valid_formation_with_captain():
    players = fake_players()
    squad = pick_squad(players)["squad"]
    lineup = pick_lineup(players, squad)
    xi = players.set_index("id").loc[lineup["xi"]]

    assert len(xi) == 11
    assert set(lineup["xi"]) <= set(squad)
    assert lineup["captain"] in lineup["xi"] and lineup["vice"] in lineup["xi"]
    assert lineup["captain"] != lineup["vice"]
    for pos, minimum in XI_MIN.items():
        assert (xi["position"] == pos).sum() >= minimum


def test_transfers_stay_within_free_transfers_plus_hits():
    players = fake_players()
    current = pick_squad(players, budget=85)["squad"]          # a weaker team
    result = pick_squad(players, budget=100, current=current, free_transfers=2)

    assert len(set(result["squad"]) - set(current)) <= 2 + result["hits"]


def test_hit_only_taken_when_worth_it():
    players = fake_players()
    current = pick_squad(players)["squad"]
    outsider = players[~players["id"].isin(current)].index[0]
    players.loc[outsider, "price"] = 4.0

    players.loc[outsider, "xp3"] = 1000                        # huge gain -> take a -4
    assert pick_squad(players, current=current, free_transfers=0)["hits"] == 1

    # Only 1 point better than the weakest player we own in that position -> don't.
    same_pos = players["id"].isin(current) & (players["position"] == players.loc[outsider, "position"])
    players.loc[outsider, "xp3"] = players.loc[same_pos, "xp3"].min() + 1
    assert pick_squad(players, current=current, free_transfers=0)["hits"] == 0


def test_unselectable_players_are_never_picked():
    players = fake_players()
    best = players["xp3"].idxmax()
    players.loc[best, "can_select"] = False                    # best player is injured

    assert players.loc[best, "id"] not in pick_squad(players)["squad"]
