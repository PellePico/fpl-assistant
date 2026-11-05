"""Pick the best squad and lineup with integer linear programming (PuLP).

Two steps, like a real manager:
1. pick_squad:  which 15 players to own, planned over the next 3 gameweeks.
                With your current team, it decides transfers (and -4 hits if worth it).
2. pick_lineup: from those 15, the best XI and captain for the NEXT gameweek.

Each player gets yes/no variables (in squad? starting? captain?). The solver
checks the combinations exactly and returns the provably best one under the rules.
"""

import pandas as pd
import pulp

SQUAD_SIZE = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}       # valid formations
XI_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3
BENCH_WEIGHT = 0.1   # bench points count a little (cover for injuries)
HIT_COST = 4         # points lost per extra transfer
MAX_HITS = 2         # never suggest more than two hits


def _lineup_model(prob, pool: pd.DataFrame, score: str, in_squad=None):
    """Add XI + captain variables and formation rules. Returns (xi, cap, points)."""
    ids = list(pool.index)
    xi = pulp.LpVariable.dicts("xi", ids, cat="Binary")
    cap = pulp.LpVariable.dicts("cap", ids, cat="Binary")
    prob += pulp.lpSum(xi.values()) == 11
    prob += pulp.lpSum(cap.values()) == 1
    for i in ids:
        prob += cap[i] <= xi[i]                       # captain must start
        if in_squad is not None:
            prob += xi[i] <= in_squad[i]              # can only start if in the squad
    for pos in SQUAD_SIZE:
        starters = pulp.lpSum(xi[i] for i in ids if pool.at[i, "position"] == pos)
        prob += starters >= XI_MIN[pos]
        prob += starters <= XI_MAX[pos]
    # Starters count once, the captain twice.
    points = pulp.lpSum(pool.at[i, score] * (xi[i] + cap[i]) for i in ids)
    return xi, cap, points


def _solve(prob):
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise ValueError(f"No valid squad found ({pulp.LpStatus[prob.status]}).")


def pick_squad(players: pd.DataFrame, budget: float = 100.0, current: list[int] | None = None,
               free_transfers: int = 1, score: str = "xp3") -> dict:
    """The best 15 players. Returns {"squad": [ids], "hits": n}.

    With `current` (your 15 ids): changes beyond `free_transfers` cost 4 points each,
    so the solver only takes a hit when the gain is bigger than the cost.
    """
    pool = players[players["can_select"]].set_index("id")
    ids = list(pool.index)

    prob = pulp.LpProblem("squad", pulp.LpMaximize)
    squad = pulp.LpVariable.dicts("squad", ids, cat="Binary")
    xi, cap, points = _lineup_model(prob, pool, score, in_squad=squad)
    bench = pulp.lpSum(pool.at[i, score] * BENCH_WEIGHT * (squad[i] - xi[i]) for i in ids)

    prob += pulp.lpSum(pool.at[i, "price"] * squad[i] for i in ids) <= budget
    for pos, n in SQUAD_SIZE.items():
        prob += pulp.lpSum(squad[i] for i in ids if pool.at[i, "position"] == pos) == n
    for club in pool["club"].unique():
        prob += pulp.lpSum(squad[i] for i in ids if pool.at[i, "club"] == club) <= MAX_PER_CLUB

    hits = 0
    if current:
        hits = pulp.LpVariable("hits", 0, MAX_HITS, cat="Integer")
        prob += pulp.lpSum(squad[i] for i in ids if i not in current) <= free_transfers + hits

    prob += points + bench - HIT_COST * hits           # the objective
    _solve(prob)
    return {"squad": [i for i in ids if squad[i].value() > 0.5],
            "hits": int(round(hits.value())) if current else 0}


def pick_lineup(players: pd.DataFrame, squad_ids: list[int], score: str = "xp_next") -> dict:
    """Best XI, captain and vice-captain for one gameweek, from a 15-man squad."""
    pool = players.set_index("id").loc[squad_ids]
    prob = pulp.LpProblem("lineup", pulp.LpMaximize)
    xi, cap, points = _lineup_model(prob, pool, score)
    prob += points
    _solve(prob)

    starters = [i for i in squad_ids if xi[i].value() > 0.5]
    captain = next(i for i in squad_ids if cap[i].value() > 0.5)
    vice = pool.loc[[i for i in starters if i != captain], score].idxmax()
    return {"xi": starters, "captain": captain, "vice": vice}
