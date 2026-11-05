"""FPL Assistant: your team, predicted points, recommended transfers.

Run with:  streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from fpl_data import load_deadline, load_my_lineup, load_season
from ml_model import predict_points
from optimizer import HIT_COST, MAX_PER_CLUB, XI_MAX, XI_MIN, pick_lineup, pick_squad
from points_model import ASSIST_POINTS, CLEAN_SHEET_POINTS, GOAL_POINTS, add_expected_points

DEFAULT_TEAM_ID = 8576760
BACKTEST_FILE = Path(__file__).parent / "models" / "backtest_results.json"
# FPL's fixture difficulty colours: (background, text)
FDR = {1: ("#257d5a", "#fff"), 2: ("#00ff86", "#111"), 3: ("#e7e7e7", "#111"),
       4: ("#ff1751", "#fff"), 5: ("#861d46", "#fff")}

st.set_page_config(page_title="FPL Assistant", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, .stMarkdown, button, input, textarea { font-family: 'Inter', sans-serif; }
footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; max-width: 1240px; }
h1 { font-size: 1.5rem !important; font-weight: 700 !important; letter-spacing: -0.02em; margin-bottom: 0 !important; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; }
.sub { color: #6b7280; font-size: 0.9rem; margin-bottom: 0.6rem; }
.label { color: #6b7280; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }

.st-key-pitch { border-radius: 12px; padding: 22px 6px 10px;
  background: repeating-linear-gradient(180deg, #1e7a3c 0 64px, #218442 64px 128px); }
.st-key-bench { background: #f3f4f6; border-radius: 12px; padding: 10px 6px; }
.card { text-align: center; line-height: 1.2; }
.shirt { position: relative; display: inline-block; }
.shirt img { width: 46px; display: block; }
.badge { position: absolute; top: -4px; right: -12px; width: 18px; height: 18px; border-radius: 50%;
  background: #111827; color: #fff; font-size: 10px; font-weight: 700; line-height: 18px; }
.new { position: absolute; top: -4px; left: -16px; background: #fff; color: #111827; font-size: 9px;
  font-weight: 700; padding: 0 4px; border-radius: 3px; }
.st-key-pitch button, .st-key-bench button { background: #fff; color: #111827; border: 0; border-radius: 4px;
  min-height: 0; padding: 1px 4px; margin-top: 2px; }
.st-key-pitch button p, .st-key-bench button p { font-size: 12px; font-weight: 600; }
.pts { font-size: 12px; font-weight: 600; color: #fff; margin: 2px 0 2px; }
.st-key-bench .pts { color: #111827; }
.chip { display: inline-block; font-size: 10px; font-weight: 600; padding: 1px 4px; border-radius: 3px; margin: 1px; }
.fixtures { display: flex; justify-content: center; gap: 2px; flex-wrap: nowrap; overflow: hidden; }
.fixtures .chip { font-size: 9px; padding: 1px 3px; margin: 0; white-space: nowrap; }
[class*="st-key-row_"] { margin-bottom: 18px; }
.playground { background: #f3f4f6; border-radius: 8px; padding: 8px 12px; font-size: 0.85rem; color: #374151; }
table.ticker { border-collapse: collapse; width: 100%; font-size: 12px; }
table.ticker td, table.ticker th { padding: 3px 4px; text-align: center; border-bottom: 1px solid #f3f4f6; }
table.ticker th { color: #6b7280; font-weight: 500; }
table.ticker td.team { text-align: left; font-weight: 600; }
table.ticker .chip { display: block; }
</style>""", unsafe_allow_html=True)


# ---------- Data (cached, so clicking around stays fast) ----------
@st.cache_data(ttl=3600)
def data():
    """Players with predicted points per gameweek.

    Predictions come from the ML model (it won the backtest). The simple formula is kept
    alongside (formula_next, next_goals, ...) because it explains where points come from.
    """
    players, games, next_gw, finished = load_season()
    p = add_expected_points(players, games, next_gw, finished)
    p["formula_next"] = p["xp_next"]
    ml = predict_points(players, games, next_gw, finished)
    p = p.drop(columns=ml.columns).merge(ml, left_on="id", right_index=True, how="left")
    p[list(ml.columns)] = p[list(ml.columns)].fillna(0)
    p["xp_next"] = p[f"gw{next_gw}"]
    p["xp3"] = p[[f"gw{next_gw + i}" for i in range(3)]].sum(axis=1)
    return p, games, next_gw, finished


@st.cache_data(ttl=3600)
def deadline(gw: int) -> str:
    return load_deadline(gw)


@st.cache_data(ttl=600)
def my_lineup(entry: int, gw: int) -> dict:
    return load_my_lineup(entry, gw)


@st.cache_data(ttl=3600)
def recommended_transfers(squad: tuple, bank: float, free: int) -> dict:
    p = data()[0]
    budget = bank + p.set_index("id").loc[list(squad), "price"].sum()
    return pick_squad(p, budget, list(squad), free)


@st.cache_data(ttl=3600)
def recommended_lineup(squad: tuple) -> dict:
    return pick_lineup(data()[0], list(squad))


P, GAMES, NEXT_GW, FINISHED = data()
BY_ID = P.set_index("id")
GWS = [int(c[2:]) for c in BY_ID.columns if c.startswith("gw")]


# ---------- Formatting helpers ----------
def name(pid) -> str:
    return BY_ID.at[pid, "name"]


def chips(team: int, gws) -> str:
    """Fixture chips in FPL's difficulty colours, e.g. 'ARS (H)'. A blank gameweek shows '–'."""
    html = []
    for gw in gws:
        games = GAMES[(GAMES["team"] == team) & (GAMES["event"] == gw)]
        if games.empty:
            html.append('<span class="chip" style="background:#d1d5db;color:#111">–</span>')
        for _, g in games.iterrows():
            bg, fg = FDR[int(g["difficulty"])]
            html.append(f'<span class="chip" style="background:{bg};color:{fg}">{g["opponent"]} ({g["venue"]})</span>')
    return "".join(html)


def option_label(pid) -> str:
    pl = BY_ID.loc[pid]
    return f"{pl['name']} · {pl['club']} · £{pl['price']}m · {pl['xp3']:.1f} pts next 3 GWs"


# ---------- Your team, kept in the session while you edit it ----------
def team() -> dict:
    return st.session_state.team


def load_team(entry: int):
    t = my_lineup(entry, FINISHED)
    st.session_state.team = {**t, "squad": list(t["squad"]), "xi": list(t["xi"]),
                             "original": list(t["squad"]), "start_bank": t["bank"], "entry": entry}


def bench(t) -> list[int]:
    return [i for i in t["squad"] if i not in t["xi"]]


def valid_xi(xi: list[int]) -> bool:
    counts = BY_ID.loc[xi, "position"].value_counts()
    return all(XI_MIN[pos] <= counts.get(pos, 0) <= XI_MAX[pos] for pos in XI_MIN)


def fix_armband(t):
    """After a sub or transfer, make sure captain and vice are both in the XI."""
    by_pts = BY_ID.loc[t["xi"], "xp_next"].sort_values(ascending=False).index
    if t["captain"] not in t["xi"]:
        t["captain"] = next(i for i in by_pts if i != t["vice"])
    if t["vice"] not in t["xi"] or t["vice"] == t["captain"]:
        t["vice"] = next(i for i in by_pts if i != t["captain"])


def substitute(a: int, b: int):
    t = team()
    out_, in_ = (a, b) if a in t["xi"] else (b, a)
    t["xi"] = [in_ if i == out_ else i for i in t["xi"]]
    fix_armband(t)


def make_transfer(out_: int, in_: int):
    t = team()
    t["squad"] = [in_ if i == out_ else i for i in t["squad"]]
    t["xi"] = [in_ if i == out_ else i for i in t["xi"]]
    t["bank"] = round(t["bank"] + BY_ID.at[out_, "price"] - BY_ID.at[in_, "price"], 1)
    fix_armband(t)


def can_buy(t, out_: int, in_: int) -> bool:
    if in_ in t["squad"] or BY_ID.at[in_, "position"] != BY_ID.at[out_, "position"]:
        return False
    if BY_ID.at[in_, "price"] > t["bank"] + BY_ID.at[out_, "price"]:
        return False
    clubs = BY_ID.loc[[i for i in t["squad"] if i != out_], "club"]
    return (clubs == BY_ID.at[in_, "club"]).sum() < MAX_PER_CLUB


def transfers_made(t) -> int:
    return len(set(t["squad"]) - set(t["original"]))


# ---------- Player details ----------
def breakdown(pl) -> pd.DataFrame:
    """Next gameweek's predicted points, split by source, with the sum written out."""
    pos = pl["position"]
    n = max(len(GAMES[(GAMES["team"] == pl["team"]) & (GAMES["event"] == NEXT_GW)]), 1)
    matches = "" if n == 1 else f" over {n} matches"
    rows = [("Plays 60+ minutes", f"{pl['next_appearance'] / 2 / n:.0%} chance × 2 pts{matches}", pl["next_appearance"]),
            ("Goals", f"{pl['next_goals'] / GOAL_POINTS[pos]:.2f} expected goals × {GOAL_POINTS[pos]} pts",
             pl["next_goals"]),
            ("Assists", f"{pl['next_assists'] / ASSIST_POINTS:.2f} expected assists × {ASSIST_POINTS} pts",
             pl["next_assists"])]
    if CLEAN_SHEET_POINTS[pos]:
        rows.append(("Clean sheet", f"{pl['next_clean_sheet'] / CLEAN_SHEET_POINTS[pos] / n:.0%} chance × "
                     f"{CLEAN_SHEET_POINTS[pos]} pts{matches}", pl["next_clean_sheet"]))
    if pos == "GKP":
        rows.append(("Saves", f"{pl['next_saves'] * 3:.1f} expected saves ÷ 3", pl["next_saves"]))
    rows.append(("Bonus", "average bonus per 90 minutes this season", pl["next_bonus"]))
    if pos in ("GKP", "DEF"):
        rows.append(("Goals conceded", f"{-pl['next_conceded'] * 2:.1f} expected against × −½ pt",
                     pl["next_conceded"]))
    df = pd.DataFrame(rows, columns=["Source", "Calculation", "Points"])
    total = pd.DataFrame([("Total", "", df["Points"].sum())], columns=df.columns)
    return pd.concat([df, total]).round({"Points": 1})


def show_player(pid: int):
    pl = BY_ID.loc[pid]
    left, right = st.columns([1, 6])
    left.image(pl["shirt"], width=64)
    right.markdown(f"### {pl['name']}\n<div class='sub'>{pl['club']} · {pl['position']} · £{pl['price']}m</div>",
                   unsafe_allow_html=True)
    if pl["news"]:
        st.warning(pl["news"])

    m = st.columns(4)
    m[0].metric(f"Predicted points GW{NEXT_GW}", f"{pl['xp_next']:.1f}")
    m[1].metric("Predicted, next 3 GWs", f"{pl['xp3']:.1f}")
    m[2].metric("Form", f"{pl['form']:.1f}", help="FPL's form: average points per match over the last 30 days")
    m[3].metric("Selected by", f"{pl['selected_by_percent']:.1f}%")

    st.markdown("<div class='label'>Fixtures and predicted points</div>", unsafe_allow_html=True)
    for col, gw in zip(st.columns(len(GWS)), GWS):
        col.markdown(f"<div class='card'><div class='sub'>GW{gw}</div>{chips(pl['team'], [gw])}"
                     f"<div style='font-weight:600;margin-top:4px'>{pl[f'gw{gw}']:.1f}</div></div>",
                     unsafe_allow_html=True)

    st.markdown(f"<div class='label' style='margin-top:14px'>Where the points come from, GW{NEXT_GW}</div>",
                unsafe_allow_html=True)
    st.dataframe(breakdown(pl), hide_index=True, width="stretch")
    st.caption(f"This breakdown uses the simple formula ({pl['formula_next']:.1f} pts). The headline "
               f"prediction ({pl['xp_next']:.1f} pts) comes from the ML model, which tested slightly "
               "more accurate. See the Model tab.")

    st.markdown("<div class='label'>Season so far</div>", unsafe_allow_html=True)
    s = st.columns(4)
    s[0].metric("Points", int(pl["total_points"]))
    s[1].metric("Minutes", int(pl["minutes"]))
    s[2].metric("Goals (xG)", f"{pl['goals_scored']} ({pl['expected_goals']:.1f})")
    s[3].metric("Assists (xA)", f"{pl['assists']} ({pl['expected_assists']:.1f})")


def team_actions(pid: int):
    """Captain, substitution and transfer controls for a player in your squad."""
    t = team()
    st.divider()
    if pid in t["xi"]:
        a, b = st.columns(2)
        if a.button("Make captain", disabled=pid == t["captain"], width="stretch"):
            if t["vice"] == pid:
                t["vice"] = t["captain"]
            t["captain"] = pid
            st.rerun()
        if b.button("Make vice-captain", disabled=pid in (t["vice"], t["captain"]), width="stretch"):
            t["vice"] = pid
            st.rerun()
        swap_with = [b for b in bench(t) if valid_xi([b if i == pid else i for i in t["xi"]])]
    else:
        swap_with = [x for x in t["xi"] if valid_xi([pid if i == x else i for i in t["xi"]])]

    sub = st.selectbox("Substitute with", swap_with, index=None, format_func=option_label,
                       placeholder="Choose a player" if swap_with else "No valid substitution")
    if st.button("Make substitution", disabled=sub is None):
        substitute(pid, sub)
        st.rerun()

    options = P[P["can_select"]].sort_values("xp3", ascending=False)["id"]
    options = [i for i in options if can_buy(t, pid, i)]
    buy = st.selectbox(f"Transfer out {name(pid)}, bring in", options, index=None, format_func=option_label,
                       placeholder=f"{len(options)} players fit your budget and club limits")
    if st.button("Confirm transfer", disabled=buy is None):
        make_transfer(pid, buy)
        st.rerun()


@st.dialog("Player details", width="large")
def player_dialog(pid: int, in_team: bool):
    show_player(pid)
    if in_team:
        team_actions(pid)


# ---------- Pitch ----------
def card(pid: int, where: str):
    t, pl = team(), BY_ID.loc[pid]
    badge = ('<span class="badge">C</span>' if pid == t["captain"]
             else '<span class="badge">V</span>' if pid == t["vice"] else "")
    new = '<span class="new">NEW</span>' if pid not in t["original"] else ""
    st.markdown(f"<div class='card'><div class='shirt'>{new}<img src='{pl['shirt']}'>{badge}</div></div>",
                unsafe_allow_html=True)
    if st.button(pl["name"], key=f"{where}_{pid}", width="stretch"):
        player_dialog(pid, in_team=True)
    st.markdown(f"<div class='card'><div class='pts'>{pl['xp_next']:.1f}</div>"
                f"<div class='fixtures'>{chips(pl['team'], range(NEXT_GW, NEXT_GW + 3))}</div></div>",
                unsafe_allow_html=True)


def pitch_row(ids: list[int], where: str, slots: int = 5):
    """A line of players, centred, with space below so rows don't touch."""
    pad = (slots - len(ids)) / 2
    with st.container(key=f"row_{where}_{ids[0]}"):
        cols = st.columns(([pad] if pad else []) + [1] * len(ids) + ([pad] if pad else []), gap="small")
        for col, pid in zip(cols[1:] if pad else cols, ids):
            with col:
                card(pid, where)


def try_player(pid: int, key: str):
    """Compare any player with your players in that position, and put them in your team."""
    t, cand = team(), BY_ID.loc[pid]
    if pid in t["squad"]:
        st.write(f"{cand['name']} is already in your team.")
        return
    same = [i for i in t["squad"] if BY_ID.at[i, "position"] == cand["position"]]
    rows = BY_ID.loc[[pid] + same, ["name", "club", "price", "xp_next", "xp3", "form"]].copy()
    rows.insert(0, "", ["Candidate"] + ["Your team"] * len(same))
    st.dataframe(rows.rename(columns={"name": "Name", "club": "Club", "price": "Price £m",
                                      "xp_next": f"Pred. GW{NEXT_GW}", "xp3": "Pred. 3 GWs",
                                      "form": "Form"}).round(1), hide_index=True, width="stretch")

    out = st.selectbox(f"Replace with {cand['name']}", same, index=None, key=f"{key}_out",
                       format_func=option_label, placeholder=f"Choose one of your {cand['position']}s")
    if out is None:
        return
    bank_after = t["bank"] + BY_ID.at[out, "price"] - cand["price"]
    gain = cand["xp3"] - BY_ID.at[out, "xp3"]
    money = f"£{bank_after:.1f}m" if bank_after >= 0 else f"−£{-bank_after:.1f}m"
    st.markdown(f"Bank after: **{money}** · predicted change: **{gain:+.1f} pts** over 3 GWs")
    clubs = BY_ID.loc[[i for i in t["squad"] if i != out], "club"]
    if bank_after < 0:
        st.warning(f"£{-bank_after:.1f}m over budget. FPL wouldn't allow this, but you can still try it here.")
    if (clubs == cand["club"]).sum() >= MAX_PER_CLUB:
        st.warning(f"That would be 4 {cand['club']} players. FPL allows max {MAX_PER_CLUB}.")
    if st.button(f"Put {cand['name']} in my team", key=f"{key}_go", type="primary"):
        make_transfer(out, pid)
        st.rerun()


# ---------- Page ----------
st.title("FPL Assistant")
st.markdown(f"<div class='sub'>Gameweek {NEXT_GW} · deadline {deadline(NEXT_GW)}</div>", unsafe_allow_html=True)
st.markdown("<div class='playground'>Planner only. Nothing you do here changes your real FPL team; "
            "refresh the page to start over from your saved team.</div>", unsafe_allow_html=True)

team_tab, players_tab, fixtures_tab, prices_tab, model_tab = st.tabs(
    ["My team", "Players", "Fixtures", "Prices", "Model"])

with team_tab:
    c1, c2, c3 = st.columns([1, 1, 2])
    entry = c1.number_input("FPL team ID", min_value=1, value=DEFAULT_TEAM_ID)
    free = c2.number_input("Free transfers this week", min_value=0, max_value=5, value=1)
    if "team" not in st.session_state or st.session_state.team["entry"] != entry:
        load_team(int(entry))
    t = team()

    made = transfers_made(t)
    hit = max(made - free, 0) * HIT_COST
    xi = BY_ID.loc[t["xi"]]
    predicted = xi["xp_next"].sum() + BY_ID.at[t["captain"], "xp_next"]
    m = st.columns(4)
    m[0].metric(f"Predicted points GW{NEXT_GW}", f"{predicted - hit:.1f}",
                help="Starting XI, captain counted twice, minus any hit")
    m[1].metric("Transfers", f"{made} of {free} free")
    m[2].metric("Hit", f"−{hit}" if hit else "0")
    m[3].metric("Bank", f"£{t['bank']:.1f}m" if t["bank"] >= 0 else f"−£{-t['bank']:.1f}m",
                help="Negative means over budget. Fine to experiment, but FPL wouldn't allow it.")

    pitch_col, side = st.columns([5, 2], gap="large")
    with pitch_col:
        with st.container(key="pitch"):
            for pos in ["GKP", "DEF", "MID", "FWD"]:
                pitch_row([i for i in t["xi"] if BY_ID.at[i, "position"] == pos], "xi")
        with st.container(key="bench"):
            pitch_row(bench(t), "bench", slots=4)
        st.caption(f"Numbers are predicted points for GW{NEXT_GW}. Click a player to captain, "
                   "substitute or transfer them.")

    with side:
        st.markdown("### Try any player")
        cand = st.selectbox("Search", BY_ID.sort_values("xp_next", ascending=False).index, index=None,
                            format_func=option_label, placeholder="Search any player", key="try_search",
                            label_visibility="collapsed")
        if cand:
            try_player(cand, "side")

        st.markdown("### Recommended transfers")
        rec = recommended_transfers(tuple(t["original"]), t["start_bank"], int(free))
        outs = [i for i in t["original"] if i not in rec["squad"]]
        ins = [i for i in rec["squad"] if i not in t["original"]]
        by_pos = lambda ids: sorted(ids, key=lambda i: BY_ID.at[i, "element_type"])
        if not outs:
            st.write("No transfer gains enough. Save it for next week.")
        for o, n in zip(by_pos(outs), by_pos(ins)):
            gain = BY_ID.at[n, "xp3"] - BY_ID.at[o, "xp3"]
            with st.container(border=True):
                st.markdown(f"**{name(o)}** → **{name(n)}**  \n"
                            f"<span class='sub'>+{gain:.1f} pts over 3 GWs · £{BY_ID.at[o, 'price']}m → "
                            f"£{BY_ID.at[n, 'price']}m</span>", unsafe_allow_html=True)
                done = o not in t["squad"] or n in t["squad"]
                blocked = not done and not can_buy(t, o, n)
                if st.button("Done" if done else "Try this transfer", key=f"rec_{o}_{n}", disabled=done or blocked,
                             help="Do the other recommended transfers first (budget or 3-per-club limit)"
                             if blocked else None):
                    make_transfer(o, n)
                    st.rerun()
        if rec["hits"]:
            st.caption(f"Includes {rec['hits']} extra transfer(s) at −{HIT_COST} each. "
                       "Recommended only because the gain is bigger than the hit.")

        st.markdown("### Recommended lineup")
        lineup = recommended_lineup(tuple(t["squad"]))
        benched = [name(i) for i in t["xi"] if i not in lineup["xi"]]
        st.markdown(f"Captain **{name(lineup['captain'])}**, vice **{name(lineup['vice'])}**")
        if benched:
            st.markdown(f"Bench {', '.join(benched)}; start "
                        f"{', '.join(name(i) for i in lineup['xi'] if i not in t['xi'])}")
        same = set(lineup["xi"]) == set(t["xi"]) and lineup["captain"] == t["captain"]
        if st.button("Use recommended lineup", disabled=same):
            t["xi"], t["captain"], t["vice"] = lineup["xi"], lineup["captain"], lineup["vice"]
            st.rerun()

        st.markdown("### ")
        if st.button("Reset to my saved team", disabled=t["squad"] == t["original"] and made == 0):
            del st.session_state.team
            st.rerun()

with players_tab:
    pick = st.selectbox("Search for a player", BY_ID.sort_values("xp_next", ascending=False).index,
                        index=None, format_func=option_label, placeholder="Type a name")
    if pick:
        with st.container(border=True):
            show_player(pick)
            st.markdown("<div class='label' style='margin-top:10px'>Try in your team</div>",
                        unsafe_allow_html=True)
            try_player(pick, "players")

    f1, f2 = st.columns(2)
    positions = f1.multiselect("Position", ["GKP", "DEF", "MID", "FWD"], default=["GKP", "DEF", "MID", "FWD"])
    top = float(P["price"].max())
    max_price = f2.slider("Max price (£m)", 3.5, top, top, 0.5)
    table = P[P["position"].isin(positions) & (P["price"] <= max_price) & (P["minutes"] > 0)]
    st.dataframe(
        table.sort_values("xp_next", ascending=False)[[
            "name", "club", "position", "price", "xp_next", "xp3", "form", "selected_by_percent",
            "total_points", "goals_scored", "assists", "expected_goals", "expected_assists", "minutes"]]
        .rename(columns={"name": "Name", "club": "Club", "position": "Pos", "price": "Price",
                         "xp_next": f"Pred. GW{NEXT_GW}", "xp3": "Pred. 3 GWs", "form": "Form",
                         "selected_by_percent": "Owned %", "total_points": "Points", "goals_scored": "Goals",
                         "assists": "Assists", "expected_goals": "xG", "expected_assists": "xA",
                         "minutes": "Minutes"}).round(1),
        hide_index=True, width="stretch", height=520)

with fixtures_tab:
    clubs = P.drop_duplicates("team").set_index("team")["club"].sort_index()
    remaining = sorted(GAMES["event"].unique())

    st.markdown("### Double gameweeks")
    per_gw = GAMES.groupby(["event", "team"]).size()
    doubles = per_gw[per_gw > 1]
    if doubles.empty:
        st.write("None scheduled yet. Doubles appear when postponed matches (usually from cup clashes) "
                 "are rescheduled, so check back after the cup rounds.")
    else:
        for gw, teams in doubles.groupby(level="event"):
            st.markdown(f"**GW{gw}**")
            for team_id in teams.index.get_level_values("team"):
                st.markdown(f"{clubs[team_id]} &nbsp; {chips(team_id, [gw])}", unsafe_allow_html=True)

    st.markdown("### Blank gameweeks")
    blanks = [(gw, [clubs[tm] for tm in clubs.index if (gw, tm) not in per_gw.index]) for gw in remaining]
    blanks = [(gw, teams) for gw, teams in blanks if teams]
    if not blanks:
        st.write("None scheduled. Every team has a match in every remaining gameweek.")
    for gw, teams in blanks:
        st.markdown(f"**GW{gw}**: {', '.join(teams)} have no match")

    st.markdown("### Fixture difficulty, next 6 gameweeks")
    st.caption("Easiest run first. Colours are FPL's own difficulty ratings.")
    window = remaining[:6]
    ease = GAMES[GAMES["event"].isin(window)].groupby("team")["difficulty"].mean().sort_values()
    head = "".join(f"<th>GW{gw}</th>" for gw in window)
    body = "".join(f"<tr><td class='team'>{clubs[tm]}</td>"
                   + "".join(f"<td>{chips(tm, [gw])}</td>" for gw in window) + "</tr>"
                   for tm in ease.index)
    st.markdown(f"<table class='ticker'><tr><th></th>{head}</tr>{body}</table>", unsafe_allow_html=True)

with prices_tab:
    def price_table(df, col):
        out = df[["name", "club", "position", "price", col]].copy()
        out[col] = out[col].map(lambda c: f"{c / 10:+.1f}")
        return out.rename(columns={"name": "Name", "club": "Club", "position": "Pos",
                                   "price": "Price £m", col: "Change £m"})

    st.markdown("### Since the last deadline")
    a, b = st.columns(2)
    a.markdown("<div class='label'>Risers</div>", unsafe_allow_html=True)
    a.dataframe(price_table(P[P["cost_change_event"] > 0].sort_values("cost_change_event", ascending=False),
                            "cost_change_event"), hide_index=True, width="stretch")
    b.markdown("<div class='label'>Fallers</div>", unsafe_allow_html=True)
    b.dataframe(price_table(P[P["cost_change_event"] < 0].sort_values("cost_change_event"),
                            "cost_change_event"), hide_index=True, width="stretch")

    st.markdown("### Transfer trends this gameweek")
    st.caption("Prices move when many managers buy or sell a player, so these hint at the next changes.")
    trends = P.assign(net=P["transfers_in_event"] - P["transfers_out_event"])
    cols = {"name": "Name", "club": "Club", "price": "Price £m", "transfers_in_event": "In",
            "transfers_out_event": "Out", "net": "Net"}
    a, b = st.columns(2)
    a.markdown("<div class='label'>Most bought, likely to rise</div>", unsafe_allow_html=True)
    a.dataframe(trends.nlargest(10, "net")[list(cols)].rename(columns=cols), hide_index=True, width="stretch")
    b.markdown("<div class='label'>Most sold, likely to fall</div>", unsafe_allow_html=True)
    b.dataframe(trends.nsmallest(10, "net")[list(cols)].rename(columns=cols), hide_index=True, width="stretch")

    st.markdown("### Since the start of the season")
    a, b = st.columns(2)
    a.markdown("<div class='label'>Biggest risers</div>", unsafe_allow_html=True)
    a.dataframe(price_table(P.nlargest(10, "cost_change_start"), "cost_change_start"),
                hide_index=True, width="stretch")
    b.markdown("<div class='label'>Biggest fallers</div>", unsafe_allow_html=True)
    b.dataframe(price_table(P.nsmallest(10, "cost_change_start"), "cost_change_start"),
                hide_index=True, width="stretch")

with model_tab:
    st.markdown("""
### How points are predicted
**The prediction** comes from a machine learning model (gradient boosting, the same family of
methods as the open-source research model OpenFPL). It was trained on every player in every match of
the 2025-26 season. For each upcoming match it looks at the player's season so far (minutes, xG, xA,
bonus, defensive actions and more, all per 90 minutes), their form, the opponent's difficulty and
home/away. The result is multiplied by the chance the player is available (injury news).

**The explanation** in each player's details uses a simple formula. It is slightly less accurate,
but it shows where points come from:

| Source | Estimate |
|---|---|
| Plays 60+ minutes | chance of playing 60+ minutes × 2 pts, from average minutes and injury status |
| Goals | expected goals (xG per 90 × minutes) × points per goal: GK 10, DEF 6, MID 5, FWD 4 |
| Assists | expected assists (xA per 90 × minutes) × 3 |
| Clean sheet | chance of conceding 0 (Poisson, from expected goals against) × GK/DEF 4, MID 1 |
| Saves | goalkeepers: expected saves ÷ 3 |
| Bonus | average bonus per 90 minutes this season |
| Goals conceded | GK/DEF: −½ point per expected goal against |

**Opponent strength:** against an easy opponent (difficulty 1–2), expected goals and assists go up
15–30%; against a hard one (4–5), they go down 15–30%. Defenders concede more against strong teams.

**Small samples:** per-90 numbers are divided by at least 270 minutes, so one goal in a 20-minute
cameo doesn't make a player look like a star.

### How the team is chosen
An optimizer (integer linear programming) checks every valid combination and returns the best one:
the 15 players with the most predicted points over the next 3 gameweeks within your budget,
2/5/5/3 positions and max 3 per club. It may only change as many players as you have free transfers,
plus at most 2 extra at −4 each, and only when the gain is bigger than the cost.
""")
    st.markdown("### How accurate is it?")
    st.markdown("Tested on gameweeks 3–6 of this season, which the model never saw in training. "
                "Each method predicted every player's points one gameweek ahead, using only data from "
                "before that gameweek. Lower error is better.")
    names = {"ml_model": "ML model (used in this app)", "formula": "Simple formula",
             "fpl_xp": "FPL's official prediction*", "last5_mean": "Average of last 5 gameweeks",
             "zero": "Predict 0 for everyone"}
    results = pd.DataFrame(json.loads(BACKTEST_FILE.read_text()))
    results = results.assign(method=results["method"].map(names)).rename(columns={
        "method": "Method", "mae": "Average error (pts)", "rmse": "RMSE",
        "mae_played": "Average error, players who played", "n": "Predictions"})
    st.dataframe(results, hide_index=True, width="stretch")
    st.caption("*FPL's numbers come from a public archive saved after each gameweek, so they may include "
               "information from after kick-off, such as late injury news. That makes the comparison "
               "unfair in FPL's favour. Among the fair comparisons, the ML model is the most accurate, "
               "but only by about 3%. The test covers just 4 early-season gameweeks.")
