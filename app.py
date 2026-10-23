import streamlit as st

import reports as R

st.set_page_config(page_title="IPL Explorer", page_icon="🏏", layout="wide")


def render(report_function, *args, **kwargs):
    """Run one report and draw its tables and charts. Bad input shows a warning, not a crash."""
    try:
        tables, figs = report_function(*args, **kwargs)
    except ValueError as error:
        st.warning(str(error))
        return
    for title, table in tables:
        st.subheader(title)
        st.dataframe(table, hide_index=True)
    left, right = st.columns(2)
    for i, fig in enumerate(figs):
        (left if i % 2 == 0 else right).pyplot(fig)


st.sidebar.title("🏏 IPL Explorer")
page = st.sidebar.radio(
    "Choose a report",
    ["Overview", "Team", "Season", "Batter", "Bowler", "Head-to-Head", "Venue"],
)
st.sidebar.caption("Data: IPL matches and ball-by-ball records, 2008 to 2026. The 2026 season is partial.")

# ---------------------------------------------------------------- Overview
if page == "Overview":
    st.title("IPL match and player analysis")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", f"{len(R.matches):,}")
    c2.metric("Seasons", len(R.years))
    c3.metric("Teams", len(R.all_teams))
    c4.metric("Venues", R.matches["venue"].nunique())

    tab_insights, tab_charts, tab_data = st.tabs(["Key insights", "Charts", "Data and cleaning"])
    with tab_insights:
        for line in R.key_insights():
            st.markdown("- " + line)
        st.caption("Every number above is computed from the data when the page loads.")
    with tab_charts:
        left, right = st.columns(2)
        for i, fig in enumerate(R.overview_figures()):
            (left if i % 2 == 0 else right).pyplot(fig)
    with tab_data:
        st.subheader("Cleaning report")
        st.dataframe(R.CLEANING_REPORT, hide_index=True)
        st.subheader("Matches table (first 10 rows)")
        st.dataframe(R.matches.head(10))
        st.subheader("Summary statistics")
        st.dataframe(R.matches[["inn1_runs", "inn2_runs", "margin"]].describe().round(2))

# ---------------------------------------------------------------- Team
elif page == "Team":
    st.title("Team report")
    team = st.selectbox("Team", R.all_teams, index=R.all_teams.index("Mumbai Indians"))
    render(R.team_report, team)

# ---------------------------------------------------------------- Season
elif page == "Season":
    st.title("Season report")
    year = st.selectbox("Season", sorted(R.years, reverse=True))
    render(R.season_report, year)

# ---------------------------------------------------------------- Batter
elif page == "Batter":
    st.title("Batter report")
    name = st.selectbox("Batter (type to search)", R.batter_names)
    first, last = st.select_slider("Seasons", options=R.years, value=(R.years[0], R.years[-1]))
    render(R.batter_report, name, first, last)

# ---------------------------------------------------------------- Bowler
elif page == "Bowler":
    st.title("Bowler report")
    name = st.selectbox("Bowler (type to search)", R.bowler_names)
    render(R.bowler_report, name)

# ---------------------------------------------------------------- Head to head
elif page == "Head-to-Head":
    st.title("Head-to-head")
    c1, c2 = st.columns(2)
    team_a = c1.selectbox("First team", R.all_teams, index=R.all_teams.index("Chennai Super Kings"))
    team_b = c2.selectbox("Second team", R.all_teams, index=R.all_teams.index("Mumbai Indians"))
    render(R.head_to_head, team_a, team_b)

# ---------------------------------------------------------------- Venue
elif page == "Venue":
    st.title("Venue report")
    place = st.selectbox("Venue or city", R.venue_options)
    render(R.venue_report, place)
