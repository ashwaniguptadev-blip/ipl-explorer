"""IPL Explorer: data loading and report functions (used by app.py)."""
from pathlib import Path
import difflib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns

# ---------- chart style ----------
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#c9c8c3", "#0b0b0b", "#52514e"
AQUA, YELLOW = "#1baf7a", "#eda100"
plt.rcParams.update({
    "figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#b9b8b2", "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
})

# ---------- load the small tables saved from Colab ----------
DATA = Path(__file__).parent / "data"
matches = pd.read_csv(DATA / "matches.csv", parse_dates=["date"])
team_innings = pd.read_csv(DATA / "team_innings.csv")
batter_inn = pd.read_csv(DATA / "batter_inn.csv")
bowler_match = pd.read_csv(DATA / "bowler_match.csv")
bowler_kinds = pd.read_csv(DATA / "bowler_kinds.csv")
season_df = pd.read_csv(DATA / "season_df.csv")
aliases = pd.read_csv(DATA / "team_aliases.txt", header=None, names=["old", "new"])
alias_map = dict(zip(aliases["old"], aliases["new"]))

# ---------- lists used by the app's dropdowns ----------
all_teams = sorted(set(matches["team1"]) | set(matches["team2"]))
years = sorted(matches["season_year"].unique().tolist())
batter_names = batter_inn.groupby("batter")["runs"].sum().sort_values(ascending=False).index.tolist()
bowler_names = bowler_match.groupby("bowler")["wickets"].sum().sort_values(ascending=False).index.tolist()
venue_options = sorted(matches["venue"].unique()) + sorted(set(matches["city"]) - {"Unknown"})

# ---------- shown on the Overview page ----------
CLEANING_REPORT = pd.DataFrame({
    "Check": ["Team names", "Venue names", "Missing city (matches)", "Missing winner (matches)",
              "Missing player of the match", "Missing city (deliveries)", "Duplicate rows", "Date column"],
    "Before": ["19 spellings", "59 spellings", "51", "25", "9", "12,397", "0", "text"],
    "After": ["15 teams", "37 venues", "0 (filled from the venue)", "0 (labelled 'No Result')",
              "0 (labelled 'Not Awarded')", "0", "0 removed", "datetime"],
})


def key_insights():
    d = matches[matches["decided"]].copy()
    d["period"] = pd.cut(d["season_year"], [2007, 2012, 2017, 2022, 2026],
                         labels=["2008-12", "2013-17", "2018-22", "2023-26"])
    toss_rate = d["toss_win_match_win"].mean() * 100
    by_dec = d.groupby("toss_decision", observed=True)["toss_win_match_win"].mean() * 100
    chase_rate = d["chasing_win"].mean() * 100
    field_share = d.groupby("period", observed=True)["toss_decision"].apply(lambda s: (s == "field").mean() * 100)
    avg_first = matches.groupby("season_year")["inn1_runs"].mean()
    big = matches[matches["inn1_runs"] >= 200].groupby("season_year").size()
    vs = d.groupby("venue").agg(n=("match_id", "count"), chase=("chasing_win", "mean"), avg=("inn1_runs", "mean"))
    vs = vs[vs["n"] >= 30]
    best_chase, worst_chase, high_venue = vs["chase"].idxmax(), vs["chase"].idxmin(), vs["avg"].idxmax()
    wins = d.groupby("winner").size().sort_values(ascending=False)
    return [
        f"**Toss:** toss winners won {toss_rate:.1f}% of {len(d)} decided matches, a small edge. "
        f"Choosing to field won {by_dec['field']:.1f}% versus {by_dec['bat']:.1f}% for choosing to bat.",
        f"**Scoring:** the average first-innings score rose from {avg_first[2008]:.0f} in 2008 to {avg_first[2024]:.0f} in 2024, "
        f"and 200+ totals went from {big.get(2022, 0)} in 2022 to {big.get(2025, 0)} in 2025.",
        f"**Strategy shift:** captains who won the toss chose to field {field_share['2008-12']:.0f}% of the time in 2008-12, "
        f"rising to {field_share['2023-26']:.0f}% in 2023-26.",
        f"**Chasing:** chasing teams won {chase_rate:.1f}% overall, but it depends on the ground: "
        f"{best_chase} {vs.loc[best_chase, 'chase'] * 100:.0f}% vs {worst_chase} {vs.loc[worst_chase, 'chase'] * 100:.0f}%. "
        f"Highest-scoring venue: {high_venue} ({vs.loc[high_venue, 'avg']:.0f} average first innings).",
        f"**Teams:** {wins.index[0]} lead with {wins.iloc[0]} wins, then {wins.index[1]} ({wins.iloc[1]}) "
        f"and {wins.index[2]} ({wins.iloc[2]}).",
    ]


def overview_figures():
    figs = []
    trend = matches.groupby("season_year")[["inn1_runs", "inn2_runs"]].mean().round(1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(trend.index, trend["inn1_runs"], color=BLUE, linewidth=2, marker="o", markersize=5, label="Batting first")
    ax.plot(trend.index, trend["inn2_runs"], color=ORANGE, linewidth=2, marker="o", markersize=5, label="Chasing")
    ax.set_title("Average innings score by season"); ax.set_xlabel("Season"); ax.set_ylabel("Average runs")
    ax.set_xticks(trend.index[::2]); ax.legend(frameon=False, loc="upper left"); ax.grid(axis="x", visible=False)
    fig.tight_layout(); figs.append(fig)

    wins = matches[matches["decided"]].groupby("winner").size().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [BLUE] + [GRAY] * (len(wins) - 1)
    bars = ax.barh(wins.index[::-1], wins.values[::-1], color=colors[::-1], height=0.65)
    ax.bar_label(bars, padding=4, color=INK); ax.set_title("Top 10 teams by IPL wins"); ax.set_xlabel("Matches won")
    ax.grid(axis="y", visible=False); fig.tight_layout(); figs.append(fig)

    toss = matches["toss_decision"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.pie(toss.values, labels=[k.title() for k in toss.index], colors=[BLUE, ORANGE], autopct="%1.1f%%",
           startangle=90, counterclock=False, wedgeprops={"edgecolor": "white", "linewidth": 2}, textprops={"color": INK})
    ax.set_title("Toss decision: bat or field?"); fig.tight_layout(); figs.append(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(matches["inn1_runs"], bins=20, color=BLUE, edgecolor="white", linewidth=1)
    mean = matches["inn1_runs"].mean()
    ax.axvline(mean, color=INK, linestyle="--", linewidth=1.2)
    ax.text(mean + 2, ax.get_ylim()[1] * 0.92, f"mean {mean:.0f}", color=INK)
    ax.set_title("Distribution of first-innings scores"); ax.set_xlabel("Runs"); ax.set_ylabel("Matches")
    fig.tight_layout(); figs.append(fig)

    runs_wins = matches[matches["margin_type"] == "runs"]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=runs_wins, x="margin", bins=25, kde=True, color=BLUE, edgecolor="white", linewidth=1, ax=ax)
    ax.set_title("Winning margins (team batting first won)"); ax.set_xlabel("Margin (runs)"); ax.set_ylabel("Matches")
    fig.tight_layout(); figs.append(fig)

    top_venues = matches["venue"].value_counts().head(8).index
    vdata = matches[matches["venue"].isin(top_venues)]
    order = vdata.groupby("venue")["inn1_runs"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=vdata, y="venue", x="inn1_runs", order=order, color=BLUE, width=0.55, linewidth=1.2, fliersize=3, ax=ax)
    for patch in ax.patches:
        patch.set_alpha(0.55)
    ax.set_title("First-innings scores at the 8 busiest venues"); ax.set_xlabel("Runs"); ax.set_ylabel("")
    ax.grid(axis="y", visible=False); fig.tight_layout(); figs.append(fig)
    for f in figs:
        plt.close(f)
    return figs


# ---------- helpers ----------
def find_team(text):
    t = alias_map.get(text.strip().lower(), text.strip())
    lower = {x.lower(): x for x in all_teams}
    if t.lower() in lower:
        return lower[t.lower()]
    hits = [x for x in all_teams if t.lower() in x.lower()]
    if len(hits) == 1:
        return hits[0]
    close = difflib.get_close_matches(t, all_teams, n=3, cutoff=0.5)
    raise ValueError(f"Team not found: '{text}'. Did you mean: {', '.join(close or hits or all_teams[:5])}?")

def find_player(text, names):
    counts = names.value_counts()                      # appearances per player
    pool = pd.Series(counts.index)
    t = text.strip().lower()
    exact = pool[pool.str.lower() == t]
    if len(exact):
        return exact.iloc[0]
    hits = pool[pool.str.lower().str.contains(t, regex=False)]
    hits = hits.sort_values(key=lambda s: s.map(counts), ascending=False)
    if len(hits) == 1:
        return hits.iloc[0]
    if len(hits) > 1:
        if counts[hits.iloc[0]] >= 5 * counts[hits.iloc[1]]:      # one clear regular player
            return hits.iloc[0]
        raise ValueError(f"Several players match '{text}': {', '.join(hits.head(5))}. Type the full name.")
    close = difflib.get_close_matches(text.strip(), list(pool), n=3, cutoff=0.5)
    raise ValueError(f"Player not found: '{text}'. Did you mean: {', '.join(close) or 'no close match'}?")

def summary_table(d):
    return pd.DataFrame({"Metric": list(d.keys()), "Value": [str(v) for v in d.values()]})

# ---------------- 1. TEAM REPORT ----------------
def team_report(team_text):
    team = find_team(team_text)
    m = matches[(matches["team1"] == team) | (matches["team2"] == team)]
    d = m[m["decided"]]
    wins = d[d["winner"] == team]
    toss_won = d[d["toss_winner"] == team]
    toss_conv = (toss_won["winner"] == team).mean() * 100
    best_venue = wins["venue"].value_counts().index[0] if len(wins) else "n/a"
    summary = summary_table({
        "Matches played": len(m), "Wins": len(wins), "Win %": round(len(wins) / len(d) * 100, 1),
        "Titles": int((season_df["champion"] == team).sum()),
        "Toss won -> match won %": round(toss_conv, 1), "Best venue (most wins)": best_venue})
    figs = []
    played = m.groupby("season_year").size()
    w = wins.groupby("season_year").size().reindex(played.index, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(w.index, w.values, color=BLUE, linewidth=2, marker="o", markersize=5)
    ax.set_title(f"{team}: wins per season"); ax.set_xlabel("Season"); ax.set_ylabel("Wins")
    ax.grid(axis="x", visible=False); fig.tight_layout(); figs.append(fig)

    bat_first = int((wins["batting_first"] == team).sum())
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.pie([bat_first, len(wins) - bat_first], labels=["Batting first", "Chasing"], colors=[BLUE, ORANGE],
           autopct="%1.1f%%", startangle=90, counterclock=False,
           wedgeprops={"edgecolor": "white", "linewidth": 2}, textprops={"color": INK})
    ax.set_title("How the wins came"); fig.tight_layout(); figs.append(fig)

    top_v = wins["venue"].value_counts().head(5)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.barh(top_v.index[::-1], top_v.values[::-1], color=BLUE, height=0.6)
    ax.bar_label(bars, padding=3, color=INK); ax.set_title("Top venues by wins"); ax.set_xlabel("Wins")
    ax.grid(axis="y", visible=False); fig.tight_layout(); figs.append(fig)

    ti = team_innings[team_innings["batting_team"] == team]
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=ti, x="season_year", y="runs", color=BLUE, width=0.6, linewidth=1, fliersize=3, ax=ax)
    for patch in ax.patches:
        patch.set_alpha(0.55)
    ax.set_title("Innings scores by season"); ax.set_xlabel("Season"); ax.set_ylabel("Runs")
    ax.tick_params(axis="x", rotation=45); ax.grid(axis="x", visible=False); fig.tight_layout(); figs.append(fig)
    for f in figs:
        plt.close(f)
    return [(f"Team report: {team}", summary)], figs

# ---------------- 2. SEASON REPORT ----------------
def season_report(year):
    year = int(year)
    m = matches[matches["season_year"] == year]
    if m.empty:
        raise ValueError(f"No matches for {year}. Choose a year from {matches['season_year'].min()} to {matches['season_year'].max()}.")
    d = m[m["decided"]]
    champ = season_df.loc[season_df["season"] == year, "champion"]
    champ = champ.iloc[0] if len(champ) else "Not recorded (season in progress)"
    top_bat = batter_inn[batter_inn["season_year"] == year].groupby("batter")["runs"].sum().nlargest(1)
    top_bowl = bowler_match[bowler_match["season_year"] == year].groupby("bowler")["wickets"].sum().nlargest(1)
    summary = summary_table({
        "Matches": len(m), "Champion": champ,
        "Most runs": f"{top_bat.index[0]} ({top_bat.iloc[0]})",
        "Most wickets": f"{top_bowl.index[0]} ({top_bowl.iloc[0]})",
        "Average first-innings score": round(m["inn1_runs"].mean(), 1),
        "Highest total": int(m["inn1_runs"].max())})
    wins_tbl = d.groupby("winner").agg(wins=("match_id", "count")).sort_values("wins", ascending=False).reset_index()
    figs = []
    colors = [BLUE if t == champ else GRAY for t in wins_tbl["winner"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(wins_tbl["winner"][::-1], wins_tbl["wins"][::-1], color=colors[::-1], height=0.65)
    ax.bar_label(bars, padding=3, color=INK); ax.set_title(f"{year}: wins per team (champion in blue)")
    ax.set_xlabel("Wins"); ax.grid(axis="y", visible=False); fig.tight_layout(); figs.append(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(team_innings[team_innings["season_year"] == year]["runs"], bins=15, color=BLUE, edgecolor="white", linewidth=1)
    ax.set_title(f"{year}: distribution of innings scores"); ax.set_xlabel("Runs"); ax.set_ylabel("Innings")
    fig.tight_layout(); figs.append(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=m, x="toss_decision", order=["field", "bat"], hue="toss_decision",
                  palette={"field": BLUE, "bat": ORANGE}, legend=False, ax=ax)
    ax.set_title(f"{year}: toss decisions"); ax.set_xlabel(""); ax.set_ylabel("Matches")
    ax.grid(axis="x", visible=False); fig.tight_layout(); figs.append(fig)
    for f in figs:
        plt.close(f)
    return [(f"Season report: {year}", summary), ("Wins per team", wins_tbl)], figs

# ---------------- 3. BATTER REPORT ----------------
def batter_report(name, from_year=None, to_year=None):
    player = find_player(name, batter_inn["batter"])
    b = batter_inn[batter_inn["batter"] == player]
    if from_year:
        b = b[b["season_year"] >= int(from_year)]
    if to_year:
        b = b[b["season_year"] <= int(to_year)]
    if b.empty:
        raise ValueError(f"{player} has no innings in that year range.")
    runs, balls, outs = b["runs"].sum(), b["balls"].sum(), b["out"].sum()
    summary = summary_table({
        "Matches": b["match_id"].nunique(), "Innings": len(b), "Runs": runs, "Balls faced": balls,
        "Strike rate": round(runs / balls * 100, 2),
        "Average": round(runs / outs, 2) if outs else "n/a (never dismissed)",
        "Highest score": int(b["runs"].max()),
        "50s": int(((b["runs"] >= 50) & (b["runs"] < 100)).sum()), "100s": int((b["runs"] >= 100).sum()),
        "Fours": int(b["fours"].sum()), "Sixes": int(b["sixes"].sum())})
    figs = []
    per_season = b.groupby("season_year")["runs"].sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(per_season.index, per_season.values, color=BLUE, linewidth=2, marker="o", markersize=5)
    ax.set_title(f"{player}: runs per season"); ax.set_xlabel("Season"); ax.set_ylabel("Runs")
    ax.grid(axis="x", visible=False); fig.tight_layout(); figs.append(fig)

    singles, fours, sixes = b["singles"].sum(), b["fours"].sum() * 4, b["sixes"].sum() * 6
    others = max(runs - singles - fours - sixes, 0)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.pie([singles, others, fours, sixes], labels=["Singles", "2s, 3s & other", "Fours", "Sixes"],
           colors=[BLUE, ORANGE, AQUA, YELLOW], autopct="%1.0f%%", startangle=90, counterclock=False,
           wedgeprops={"edgecolor": "white", "linewidth": 2}, textprops={"color": INK})
    ax.set_title("Where the runs come from"); fig.tight_layout(); figs.append(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(data=b, x="balls", y="runs", color=BLUE, s=35, edgecolor="white", linewidth=0.6, ax=ax)
    ax.set_title("Runs vs balls faced (each dot = one innings)"); ax.set_xlabel("Balls faced"); ax.set_ylabel("Runs")
    fig.tight_layout(); figs.append(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=b, x="runs", bins=12, kde=len(b) >= 10, color=BLUE, edgecolor="white", linewidth=1, ax=ax)
    ax.set_title("Distribution of innings scores"); ax.set_xlabel("Runs in an innings"); ax.set_ylabel("Innings")
    fig.tight_layout(); figs.append(fig)
    for f in figs:
        plt.close(f)
    return [(f"Batter report: {player}", summary)], figs

# ---------------- 4. BOWLER REPORT ----------------
def bowler_report(name):
    player = find_player(name, bowler_match["bowler"])
    w = bowler_match[bowler_match["bowler"] == player]
    wickets, balls, runs = w["wickets"].sum(), w["balls"].sum(), w["runs"].sum()
    best = w.sort_values(["wickets", "runs"], ascending=[False, True]).iloc[0]
    summary = summary_table({
        "Matches": len(w), "Wickets": wickets, "Overs bowled": f"{balls // 6}.{balls % 6}",
        "Runs conceded": runs, "Economy": round(runs / (balls / 6), 2),
        "Average": round(runs / wickets, 2) if wickets else "n/a",
        "Strike rate (balls/wkt)": round(balls / wickets, 1) if wickets else "n/a",
        "Best figures": f"{int(best['wickets'])}/{int(best['runs'])}"})
    figs = []
    per_season = w.groupby("season_year")["wickets"].sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(per_season.index, per_season.values, color=BLUE, width=0.6)
    ax.set_title(f"{player}: wickets per season"); ax.set_xlabel("Season"); ax.set_ylabel("Wickets")
    ax.grid(axis="x", visible=False); fig.tight_layout(); figs.append(fig)

    kinds = bowler_kinds[bowler_kinds["bowler"] == player].set_index("wicket_kind")["count"].sort_values(ascending=False)
    if len(kinds) > 3:
        kinds = pd.concat([kinds.head(3), pd.Series({"other": kinds.iloc[3:].sum()})])
    if kinds.sum() > 0:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.pie(kinds.values, labels=kinds.index, colors=[BLUE, ORANGE, AQUA, GRAY][:len(kinds)],
               autopct="%1.0f%%", startangle=90, counterclock=False,
               wedgeprops={"edgecolor": "white", "linewidth": 2}, textprops={"color": INK})
        ax.set_title("How the wickets came"); fig.tight_layout(); figs.append(fig)

    fig, ax = plt.subplots(figsize=(8, 3))
    sns.boxplot(x=w["runs"], color=BLUE, width=0.4, linewidth=1.2, fliersize=3, ax=ax)
    for patch in ax.patches:
        patch.set_alpha(0.55)
    ax.set_title("Runs conceded per match"); ax.set_xlabel("Runs"); ax.grid(axis="y", visible=False)
    fig.tight_layout(); figs.append(fig)
    for f in figs:
        plt.close(f)
    return [(f"Bowler report: {player}", summary)], figs

# ---------------- 5. HEAD TO HEAD ----------------
def head_to_head(team_a, team_b):
    A, B = find_team(team_a), find_team(team_b)
    if A == B:
        raise ValueError("Please enter two different teams.")
    m = matches[((matches["team1"] == A) & (matches["team2"] == B)) |
                ((matches["team1"] == B) & (matches["team2"] == A))].sort_values("date")
    if m.empty:
        raise ValueError(f"{A} and {B} have never played each other in this data.")
    wa, wb, nr = int((m["winner"] == A).sum()), int((m["winner"] == B).sum()), int((~m["decided"]).sum())
    summary = summary_table({"Meetings": len(m), f"{A} wins": wa, f"{B} wins": wb, "No result / tie": nr})
    last5 = m.tail(5).iloc[::-1][["date", "venue", "winner", "margin", "margin_type"]].copy()
    last5["date"] = last5["date"].dt.date
    figs = []
    labels, vals, cols = [A, B], [wa, wb], [BLUE, ORANGE]
    if nr:
        labels.append("No result"); vals.append(nr); cols.append(GRAY)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.pie(vals, labels=labels, colors=cols, autopct="%1.0f%%", startangle=90, counterclock=False,
           wedgeprops={"edgecolor": "white", "linewidth": 2}, textprops={"color": INK})
    ax.set_title("Win split"); fig.tight_layout(); figs.append(fig)

    years = sorted(m["season_year"].unique())
    fig, ax = plt.subplots(figsize=(8, 4))
    for team, col in [(A, BLUE), (B, ORANGE)]:
        s = m[m["winner"] == team].groupby("season_year").size().reindex(years, fill_value=0)
        ax.plot(s.index, s.values, color=col, linewidth=2, marker="o", markersize=5, label=team)
    ax.set_title("Wins in this fixture by year"); ax.set_xlabel("Season"); ax.set_ylabel("Wins")
    ax.set_xticks(years); ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False); ax.grid(axis="x", visible=False); fig.tight_layout(); figs.append(fig)

    ti = team_innings[team_innings["match_id"].isin(m["match_id"]) & team_innings["batting_team"].isin([A, B])]
    avg = ti.groupby("batting_team")["runs"].mean().reindex([A, B]).round(1)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(avg.index, avg.values, color=[BLUE, ORANGE], width=0.5)
    ax.bar_label(bars, padding=3, color=INK); ax.set_title("Average innings score in this fixture")
    ax.set_ylabel("Runs"); ax.grid(axis="x", visible=False); fig.tight_layout(); figs.append(fig)
    for f in figs:
        plt.close(f)
    return [(f"Head-to-head: {A} vs {B}", summary), ("Last 5 meetings (newest first)", last5)], figs

# ---------------- 6. VENUE REPORT ----------------
def venue_report(text):
    t = text.strip().lower()
    mask = matches["venue"].str.lower().str.contains(t, regex=False) | matches["city"].str.lower().str.contains(t, regex=False)
    m = matches[mask]
    if m.empty:
        pool = sorted(set(matches["venue"]) | set(matches["city"]))
        close = difflib.get_close_matches(text.strip(), pool, n=3, cutoff=0.4)
        raise ValueError(f"Venue or city not found: '{text}'. Did you mean: {', '.join(close) or 'no close match'}?")
    d = m[m["decided"]]
    summary = summary_table({
        "Grounds matched": ", ".join(sorted(m["venue"].unique())), "Matches hosted": len(m),
        "Average first-innings score": round(m["inn1_runs"].mean(), 1),
        "Highest total": int(m["inn1_runs"].max()),
        "Chasing win %": round(d["chasing_win"].mean() * 100, 1),
        "Toss winners who chose to field %": round((m["toss_decision"] == "field").mean() * 100, 1)})
    figs = []
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(m["inn1_runs"], bins=15, color=BLUE, edgecolor="white", linewidth=1)
    ax.set_title("First-innings scores here"); ax.set_xlabel("Runs"); ax.set_ylabel("Matches")
    fig.tight_layout(); figs.append(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=m, x="toss_decision", order=["field", "bat"], hue="toss_decision",
                  palette={"field": BLUE, "bat": ORANGE}, legend=False, ax=ax)
    ax.set_title("Toss decisions here"); ax.set_xlabel(""); ax.set_ylabel("Matches")
    ax.grid(axis="x", visible=False); fig.tight_layout(); figs.append(fig)

    top_t = d["winner"].value_counts().head(5)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.barh(top_t.index[::-1], top_t.values[::-1], color=BLUE, height=0.6)
    ax.bar_label(bars, padding=3, color=INK); ax.set_title("Most wins at this venue"); ax.set_xlabel("Wins")
    ax.grid(axis="y", visible=False); fig.tight_layout(); figs.append(fig)
    for f in figs:
        plt.close(f)
    return [(f"Venue report: {text}", summary)], figs
