# governance_prototype/dashboard_streamlit.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="AI-Governance Prototype", layout="wide")

# ---------- Ministry routing map ----------
MINISTRY_MAP = {
    "Health":    "Public Health",
    "Roads":     "Public Works",
    "Water":     "Water Supply & Sanitation",
    "Education": "School Education & Sports",
    "Safety":    "Urban Development",  # adjust if Safety -> Home
}
MINISTRY_OPTIONS = sorted(set(MINISTRY_MAP.values()) | {"Cross-Ministry"})

# ---------- Plain-English glossary ----------
GLOSSARY = {
    "pred_sla_breach_prob": ("Chance of missing SLA", "Model-estimated probability that this ticket will miss its service-level deadline."),
    "sla_breach": ("Observed breach rate", "Share of tickets that actually missed their SLA in the filtered set."),
    "priority_score": ("Priority score", "Baseline priority from fixed weights on risk, severity, vulnerable HHs, sentiment, and festival."),
    "priority_dynamic": ("Priority*", "Priority recomputed from the sliders in Priority Queue. Range 0–100."),
    "severity": ("Severity", "Impact level: 1=low, 2=medium, 3=high."),
    "citizen_sentiment": ("Citizen sentiment", "Public sentiment about the issue. −1=very negative, +1=very positive."),
    "vulnerable_population": ("Vulnerable households", "Flag indicating vulnerable households are impacted."),
    "festival_season": ("Festival season", "Flag indicating period overlaps key festivals."),
    "historical_backlog": ("Backlog", "Open pending tickets historically in the area."),
    "resource_availability": ("Staff available", "Relative staffing/resources available."),
    "importance_mean": ("Feature importance", "Average change in model error when this feature is shuffled."),
    "importance_std": ("Importance variability", "Variability of the importance across folds/shuffles."),
}

def label(term: str) -> str:
    return GLOSSARY.get(term, (term.replace("_", " ").title(), ""))[0]

def help_text(term: str) -> str:
    return GLOSSARY.get(term, ("", ""))[1]

def hover(term: str) -> str:
    txt, tip = label(term), help_text(term)
    return f"<span class='gloss' title='{tip}'>{txt}</span>" if tip else txt

# Hover styling for terms shown via st.markdown(..., unsafe_allow_html=True)
st.markdown(
    """
<style>
.gloss { border-bottom: 1px dotted #999; cursor: help; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Data loading ----------
@st.cache_data
def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["pred_sla_breach_prob"] = pd.to_numeric(df["pred_sla_breach_prob"], errors="coerce").clip(0, 1).fillna(0.0)
    df["priority_score"] = pd.to_numeric(df["priority_score"], errors="coerce").clip(0, 100).fillna(0.0)
    return df

@st.cache_data
def load_feature_importance(path: str) -> pd.DataFrame:
    fi = pd.read_csv(path)
    return fi.sort_values("importance_mean", ascending=False)

def compute_dynamic_priority(df: pd.DataFrame, w: dict) -> pd.Series:
    ws = np.array([w["risk"], w["severity"], w["vuln"], w["sent"], w["fest"]], dtype=float)
    ws = ws / ws.sum() if ws.sum() > 0 else ws
    risk, sev, vuln, sent, fest = ws
    pr = (
        risk * df["pred_sla_breach_prob"]
        + sev  * (df["severity"] / 3.0)
        + vuln * (df["vulnerable_population"].astype(float))
        + sent * ((-df["citizen_sentiment"] + 1) / 2.0)
        + fest * (df["festival_season"].astype(float))
    )
    return (100 * np.clip(pr, 0, 1)).round(1)

DATA_PATH = str( "governance_prototype/predictions.csv")
FI_PATH   = str( "governance_prototype/feature_importance.csv")

df = load_predictions(DATA_PATH)
fi = load_feature_importance(FI_PATH)

# ---------- Sidebar filters ----------
st.sidebar.title("Filters")
min_date, max_date = df["created_at"].min().date(), df["created_at"].max().date()
date_range = st.sidebar.slider("Created between", min_value=min_date, max_value=max_date, value=(min_date, max_date))

districts = st.sidebar.multiselect("Districts", sorted(df["district"].dropna().unique()), placeholder="All")
depts     = st.sidebar.multiselect("Departments", sorted(df["dept"].dropna().unique()), placeholder="All")
wards     = st.sidebar.multiselect("Wards", sorted(df["ward"].dropna().unique()), placeholder="All")

severity_min       = st.sidebar.select_slider(label("severity"), options=[1, 2, 3], value=1, help=help_text("severity"))
prob_min, prob_max = st.sidebar.slider(label("pred_sla_breach_prob"), 0.0, 1.0, (0.0, 1.0), 0.01, help=help_text("pred_sla_breach_prob"))
prio_min, prio_max = st.sidebar.slider(label("priority_score"), 0.0, 100.0, (0.0, 100.0), 1.0, help=help_text("priority_score"))
search_text        = st.sidebar.text_input("Search (ticket/category/ward)", "", help="Type to filter across ticket id, category, or ward.")

mask = (
    (df["created_at"].dt.date >= date_range[0])
    & (df["created_at"].dt.date <= date_range[1])
    & (df["severity"] >= severity_min)
    & (df["pred_sla_breach_prob"].between(prob_min, prob_max))
    & (df["priority_score"].between(prio_min, prio_max))
)
if districts: mask &= df["district"].isin(districts)
if depts:     mask &= df["dept"].isin(depts)
if wards:     mask &= df["ward"].isin(wards)
if search_text.strip():
    s = search_text.strip().lower()
    mask &= (
        df["ticket_id"].fillna("").str.lower().str.contains(s, na=False)
        | df["category"].fillna("").str.lower().str.contains(s, na=False)
        | df["ward"].fillna("").str.lower().str.contains(s, na=False)
    )

with st.sidebar.expander("Glossary"):
    for _k, (lbl, tip) in GLOSSARY.items():
        if tip:
            st.markdown(f"- **{lbl}** — {tip}")

fdf = df[mask].copy()

# ---------- Header KPIs ----------
st.title("AI-Powered Governance: Risk, Trends, and Triage")

k1, k2, k3, k4, k5 = st.columns(5)

# Tickets
k1.markdown("<span class='gloss' title='Count of tickets in the filtered data.'>Tickets</span>", unsafe_allow_html=True)
k1.metric("", f"{len(fdf):,}")

# Mean predicted risk
k2.markdown(hover("pred_sla_breach_prob"), unsafe_allow_html=True)
k2.metric("", f"{fdf['pred_sla_breach_prob'].mean():.2f}" if len(fdf) else "0.00")

# Observed breach rate
k3.markdown(hover("sla_breach"), unsafe_allow_html=True)
k3.metric("", f"{fdf['sla_breach'].mean():.2f}" if len(fdf) else "0.00")

# Avg priority (baseline)
k4.markdown(hover("priority_score"), unsafe_allow_html=True)
k4.metric("", f"{fdf['priority_score'].mean():.1f}" if len(fdf) else "0.0")

# Avg backlog
k5.markdown(hover("historical_backlog"), unsafe_allow_html=True)
k5.metric("", f"{fdf['historical_backlog'].mean():.0f}" if len(fdf) else "0")

st.caption("Default priority = 0.5·P(breach) + 0.2·severity + 0.15·vulnerable + 0.1·(−sent→positive) + 0.05·festival.")

if fdf.empty:
    st.warning("No records match the current filters.")
    st.stop()

# ---------- Tabs ----------
tab_overview, tab_trends, tab_drivers, tab_queue = st.tabs(["Overview", "Trends", "Drivers", "Priority Queue"])

# ====== Overview ======
with tab_overview:
    c1, c2 = st.columns(2)

    risk_by_dist = (
        fdf.groupby("district", as_index=False)["pred_sla_breach_prob"]
        .mean().sort_values("pred_sla_breach_prob", ascending=False)
    )
    fig1 = px.bar(
        risk_by_dist,
        x="district",
        y="pred_sla_breach_prob",
        title="Risk by District",
        labels={"pred_sla_breach_prob": label("pred_sla_breach_prob"), "district": "District"},
    )
    c1.plotly_chart(fig1, use_container_width=True)

    risk_by_dept = (
        fdf.groupby("dept", as_index=False)["pred_sla_breach_prob"]
        .mean().sort_values("pred_sla_breach_prob", ascending=False)
    )
    fig2 = px.bar(
        risk_by_dept,
        x="pred_sla_breach_prob",
        y="dept",
        orientation="h",
        title="Risk by Department",
        labels={"pred_sla_breach_prob": label("pred_sla_breach_prob"), "dept": "Department"},
    )
    c2.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    by_cat = (
        fdf.groupby(["dept", "category"], as_index=False)
        .agg(tickets=("ticket_id", "count"), risk=("pred_sla_breach_prob", "mean"))
        .sort_values(["risk", "tickets"], ascending=[False, False]).head(15)
    )
    fig3 = px.scatter(
        by_cat,
        x="risk",
        y="category",
        size="tickets",
        color="dept",
        title="Top categories by risk and volume",
        labels={"risk": label("pred_sla_breach_prob"), "tickets": "Ticket count", "category": "Category", "dept": "Department"},
    )
    c3.plotly_chart(fig3, use_container_width=True)

    fig4 = px.histogram(
        fdf,
        x="pred_sla_breach_prob",
        nbins=30,
        title="Risk distribution",
        histnorm="percent",
        labels={"pred_sla_breach_prob": label("pred_sla_breach_prob")},
    )
    c4.plotly_chart(fig4, use_container_width=True)

# ====== Trends ======
with tab_trends:
    fdf["day"] = fdf["created_at"].dt.date
    daily = (
        fdf.groupby("day", as_index=False)
        .agg(tickets=("ticket_id", "count"), risk=("pred_sla_breach_prob", "mean"), breaches=("sla_breach", "mean"))
        .sort_values("day")
    )

    t1, t2 = st.columns(2)
    fig5 = px.line(daily, x="day", y="tickets", title="Daily ticket volume", labels={"day": "Date", "tickets": "Tickets"})
    t1.plotly_chart(fig5, use_container_width=True)

    fig6 = px.line(
        daily,
        x="day",
        y=["risk", "breaches"],
        title="Predicted risk vs observed breach rate",
        labels={"day": "Date", "value": "Rate", "variable": "Series"},
    )
    t2.plotly_chart(fig6, use_container_width=True)

    fdf["week"] = fdf["created_at"].dt.to_period("W").dt.start_time
    weekly_dept = (
        fdf.groupby(["week", "dept"], as_index=False)["ticket_id"]
        .count().rename(columns={"ticket_id": "tickets"}).sort_values("week")
    )
    fig7 = px.area(
        weekly_dept,
        x="week",
        y="tickets",
        color="dept",
        title="Weekly tickets by department",
        labels={"week": "Week start", "tickets": "Tickets", "dept": "Department"},
    )
    st.plotly_chart(fig7, use_container_width=True)

# ====== Drivers ======
with tab_drivers:
    left, right = st.columns([1, 1])

    fig8 = px.bar(
        fi.head(15),
        x="importance_mean",
        y="feature",
        orientation="h",
        error_x="importance_std",
        title=label("importance_mean"),
        labels={
            "importance_mean": label("importance_mean"),
            "feature": "Feature",
            "importance_std": label("importance_std"),
        },
    )
    left.plotly_chart(fig8, use_container_width=True)

    df_bins = fdf.copy()
    df_bins["sentiment_bin"] = pd.qcut(df_bins["citizen_sentiment"], q=10, duplicates="drop")
    df_bins["backlog_bin"]   = pd.qcut(df_bins["historical_backlog"], q=10, duplicates="drop")
    df_bins["resource_bin"]  = pd.qcut(df_bins["resource_availability"], q=10, duplicates="drop")

    feat_options = {
        "severity": (label("severity"), help_text("severity")),
        "sentiment_bin": ("Citizen sentiment (binned)", help_text("citizen_sentiment")),
        "backlog_bin": ("Backlog (binned)", help_text("historical_backlog")),
        "resource_bin": ("Staff available (binned)", help_text("resource_availability")),
        "festival_season": (label("festival_season"), help_text("festival_season")),
    }
    feature = right.selectbox(
        "Univariate driver view",
        list(feat_options.keys()),
        index=0,
        format_func=lambda k: feat_options[k][0],
        help="Hover axis labels for meanings; see Glossary in sidebar.",
    )

    view = df_bins.groupby(feature, as_index=False)["pred_sla_breach_prob"].mean()
    # Convert Interval bins to ordered strings
    if pd.api.types.is_interval_dtype(view[feature]) or (
        pd.api.types.is_categorical_dtype(view[feature]) and
        pd.api.types.is_interval_dtype(view[feature].cat.categories)
    ):
        view["__bin_left"] = view[feature].apply(lambda iv: float(iv.left))
        view = view.sort_values("__bin_left")
        view["__label"] = view[feature].astype(str)
        x_col = "__label"
    else:
        x_col = feature
    fig9 = px.bar(
    view,
    x=x_col,
    y="pred_sla_breach_prob",
    title="Risk by " + feat_options.get(feature, (feature, ""))[0],)
    right.plotly_chart(fig9, use_container_width=True)

# ====== Priority Queue (interactive ministry routing) ======
with tab_queue:
    st.subheader("Interactive Priority Queue")

    # What-if weights
    c1, c2, c3, c4, c5 = st.columns(5)
    w = {
        "risk":    c1.slider(label("pred_sla_breach_prob"), 0.0, 1.0, 0.50, 0.01, help=help_text("pred_sla_breach_prob")),
        "severity":c2.slider(label("severity"),             0.0, 1.0, 0.20, 0.01, help=help_text("severity")),
        "vuln":    c3.slider(label("vulnerable_population"),0.0, 1.0, 0.15, 0.01, help=help_text("vulnerable_population")),
        "sent":    c4.slider(label("citizen_sentiment"),    0.0, 1.0, 0.10, 0.01, help=help_text("citizen_sentiment")),
        "fest":    c5.slider(label("festival_season"),      0.0, 1.0, 0.05, 0.01, help=help_text("festival_season")),
    }

    fdf = fdf.copy()
    fdf["priority_dynamic"] = compute_dynamic_priority(fdf, w)
    fdf["ministry_default"] = fdf["dept"].map(MINISTRY_MAP).fillna("Cross-Ministry")

    # Capacity planning and previews
    with st.expander("Capacity planning and bulk routing"):
        st.caption("Per-department capacity")
        dept_cols = st.columns(5)
        dept_list = sorted(fdf["dept"].unique())
        dept_cap = {}
        for i, d in enumerate(dept_list):
            dept_cap[d] = dept_cols[i % 5].number_input(f"{d}", min_value=0, value=50, step=10)

        st.caption("Per-ministry capacity")
        min_cols = st.columns(5)
        min_list = sorted(set(MINISTRY_MAP.values()))
        min_cap = {}
        for i, m in enumerate(min_list):
            min_cap[m] = min_cols[i % 5].number_input(f"{m}", min_value=0, value=150, step=10, key=f"cap_{m}")

        # Previews
        alloc_dept = []
        for d in dept_list:
            sub = fdf[fdf["dept"] == d].sort_values(["priority_dynamic", "pred_sla_breach_prob"], ascending=False)
            alloc_dept.append(sub.head(int(dept_cap[d])))
        alloc_dept = pd.concat(alloc_dept) if alloc_dept else fdf.head(0)

        alloc_min = []
        for m in min_list:
            sub = fdf[fdf["ministry_default"] == m].sort_values(["priority_dynamic", "pred_sla_breach_prob"], ascending=False)
            alloc_min.append(sub.head(int(min_cap[m])))
        alloc_min = pd.concat(alloc_min) if alloc_min else fdf.head(0)

        tA, tB = st.columns(2)
        tA.caption("Dept allocation preview")
        tA.dataframe(
            alloc_dept[["ticket_id", "dept", "district", "priority_dynamic", "pred_sla_breach_prob", "severity"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "dept": "Department",
                "district": "District",
                "priority_dynamic": st.column_config.NumberColumn(label("priority_dynamic"), format="%.1f", help=help_text("priority_dynamic")),
                "pred_sla_breach_prob": st.column_config.NumberColumn(label("pred_sla_breach_prob"), format="%.2f", help=help_text("pred_sla_breach_prob")),
                "severity": st.column_config.NumberColumn(label("severity"), help=help_text("severity")),
            },
        )
        tB.caption("Ministry allocation preview")
        tB.dataframe(
            alloc_min[["ticket_id", "ministry_default", "district", "priority_dynamic", "pred_sla_breach_prob", "severity"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "ministry_default": "Ministry",
                "district": "District",
                "priority_dynamic": st.column_config.NumberColumn(label("priority_dynamic"), format="%.1f", help=help_text("priority_dynamic")),
                "pred_sla_breach_prob": st.column_config.NumberColumn(label("pred_sla_breach_prob"), format="%.2f", help=help_text("pred_sla_breach_prob")),
                "severity": st.column_config.NumberColumn(label("severity"), help=help_text("severity")),
            },
        )

    # Editable queue with ministry routing
    editable = fdf.sort_values(["priority_dynamic", "pred_sla_breach_prob"], ascending=False).copy()
    editable.insert(0, "select", False)
    editable.insert(1, "escalate", False)
    editable.insert(2, "route_dept", editable["dept"])
    editable.insert(3, "route_ministry", editable["ministry_default"])
    editable["vulnerable_population"] = editable["vulnerable_population"].astype(bool)

    dept_options = sorted(editable["dept"].unique().tolist()) + ["Cross-Dept"]

    edited = st.data_editor(
        editable[
            [
                "select",
                "escalate",
                "route_dept",
                "route_ministry",
                "ticket_id",
                "created_at",
                "district",
                "ward",
                "dept",
                "category",
                "severity",
                "vulnerable_population",
                "pred_sla_breach_prob",
                "priority_dynamic",
                "historical_backlog",
                "reported_via",
            ]
        ],
        hide_index=True,
        use_container_width=True,
        column_config={
            "select": st.column_config.CheckboxColumn("Select"),
            "escalate": st.column_config.CheckboxColumn("Escalate"),
            "route_dept": st.column_config.SelectboxColumn("Route Dept", options=dept_options),
            "route_ministry": st.column_config.SelectboxColumn("Route Ministry", options=MINISTRY_OPTIONS),
            "created_at": st.column_config.DatetimeColumn("Created at"),
            "dept": "Department",
            "district": "District",
            "ward": "Ward",
            "category": "Category",
            "pred_sla_breach_prob": st.column_config.NumberColumn(label("pred_sla_breach_prob"), format="%.2f", help=help_text("pred_sla_breach_prob")),
            "priority_dynamic": st.column_config.NumberColumn(label("priority_dynamic"), format="%.1f", help=help_text("priority_dynamic")),
            "severity": st.column_config.NumberColumn(label("severity"), help=help_text("severity")),
            "vulnerable_population": st.column_config.CheckboxColumn(label("vulnerable_population"), help=help_text("vulnerable_population")),
            "historical_backlog": st.column_config.NumberColumn(label("historical_backlog"), help=help_text("historical_backlog")),
        },
        disabled=[
            "ticket_id",
            "created_at",
            "district",
            "ward",
            "dept",
            "category",
            "pred_sla_breach_prob",
            "priority_dynamic",
            "historical_backlog",
            "reported_via",
            "severity",
            "vulnerable_population",
        ],
        key="queue_editor",
    )

    # Selected rows
    sel = edited[edited["select"]]
    st.caption(f"Selected: {len(sel)}")

    # Build assignment payload
    def build_payload(rows: pd.DataFrame) -> list[dict]:
        return [
            {
                "ticket_id": r["ticket_id"],
                "assign": {
                    "ministry": r["route_ministry"],
                    "department": r["route_dept"],
                    "escalate": bool(r["escalate"]),
                },
                "priority_dynamic": float(r["priority_dynamic"]),
                "pred_sla_breach_prob": float(r["pred_sla_breach_prob"]),
                "metadata": {
                    "district": r["district"],
                    "ward": r["ward"],
                    "category": r["category"],
                    "severity": int(r["severity"]),
                    "vulnerable_population": bool(r["vulnerable_population"]),
                },
            }
            for _, r in rows.iterrows()
        ]

    if "assignments" not in st.session_state:
        st.session_state["assignments"] = []

    cA, cB, cC = st.columns([1, 1, 2])

    # Export selected CSV
    cA.download_button(
        "Export selected CSV",
        sel.drop(columns=["select"]).to_csv(index=False).encode("utf-8"),
        file_name="selected_queue.csv",
        mime="text/csv",
        disabled=len(sel) == 0,
    )

    # Export selected JSONL
    payload = build_payload(sel)
    jsonl = "\n".join([json.dumps(p, ensure_ascii=False) for p in payload]).encode("utf-8")
    cB.download_button(
        "Export selected JSONL",
        data=jsonl,
        file_name="selected_queue.jsonl",
        mime="application/json",
        disabled=len(sel) == 0,
    )

    # Assign selected
    if cC.button("Assign selected"):
        if len(sel) > 0:
            st.session_state["assignments"].extend(payload)
            st.success(f"Assigned {len(sel)} tickets")
        else:
            st.info("Nothing selected")

    with st.expander("Assignment log"):
        if st.session_state["assignments"]:
            log_df = pd.DataFrame(st.session_state["assignments"]).iloc[::-1].head(300)
            st.dataframe(log_df, hide_index=True, use_container_width=True)
        else:
            st.caption("No assignments yet.")

    st.caption("Priority* is recomputed from sliders. Use Route Dept and Route Ministry for correct authority targeting.")
