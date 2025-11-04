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

# ---------- Data loading ----------
@st.cache_data
def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["pred_sla_breach_prob"] = df["pred_sla_breach_prob"].clip(0, 1)
    df["priority_score"] = df["priority_score"].clip(0, 100)
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

BASE_DIR = Path(__file__).parent
DATA_PATH = str("governance_prototype/predictions.csv")
FI_PATH   = str("governance_prototype/feature_importance.csv")

df = load_predictions(DATA_PATH)
fi = load_feature_importance(FI_PATH)

# ---------- Sidebar filters ----------
st.sidebar.title("Filters")
min_date, max_date = df["created_at"].min().date(), df["created_at"].max().date()
date_range = st.sidebar.slider("Created between", min_value=min_date, max_value=max_date, value=(min_date, max_date))

districts = st.sidebar.multiselect("Districts", sorted(df["district"].unique()), placeholder="All")
depts     = st.sidebar.multiselect("Departments", sorted(df["dept"].unique()), placeholder="All")
wards     = st.sidebar.multiselect("Wards", sorted(df["ward"].unique()), placeholder="All")

severity_min       = st.sidebar.select_slider("Min severity", options=[1, 2, 3], value=1)
prob_min, prob_max = st.sidebar.slider("Predicted breach prob", 0.0, 1.0, (0.0, 1.0), 0.01)
prio_min, prio_max = st.sidebar.slider("Priority score", 0.0, 100.0, (0.0, 100.0), 1.0)
search_text        = st.sidebar.text_input("Search (ticket/category/ward)", "")

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
        df["ticket_id"].str.lower().str.contains(s)
        | df["category"].str.lower().str.contains(s)
        | df["ward"].str.lower().str.contains(s)
    )

fdf = df[mask].copy()

# ---------- Header KPIs ----------
st.title("AI-Powered Governance: Risk, Trends, and Triage")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Tickets", f"{len(fdf):,}")
k2.metric("Mean predicted risk", f"{fdf['pred_sla_breach_prob'].mean():.2f}" if len(fdf) else "0.00")
k3.metric("Observed breach rate", f"{fdf['sla_breach'].mean():.2f}" if len(fdf) else "0.00")
k4.metric("Avg priority", f"{fdf['priority_score'].mean():.1f}" if len(fdf) else "0.0")
k5.metric("Avg backlog", f"{fdf['historical_backlog'].mean():.0f}" if len(fdf) else "0")

# st.caption("Default priority = 0.5·P(breach) + 0.2·severity + 0.15·vulnerable + 0.1·(−sent→positive) + 0.05·festival.")

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
    fig1 = px.bar(risk_by_dist, x="district", y="pred_sla_breach_prob",
                  title="Risk by District", labels={"pred_sla_breach_prob": "Mean predicted breach"})
    c1.plotly_chart(fig1, use_container_width=True)

    risk_by_dept = (
        fdf.groupby("dept", as_index=False)["pred_sla_breach_prob"]
        .mean().sort_values("pred_sla_breach_prob", ascending=False)
    )
    fig2 = px.bar(risk_by_dept, x="pred_sla_breach_prob", y="dept", orientation="h",
                  title="Risk by Department", labels={"pred_sla_breach_prob": "Mean predicted breach"})
    c2.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    by_cat = (
        fdf.groupby(["dept", "category"], as_index=False)
        .agg(tickets=("ticket_id", "count"), risk=("pred_sla_breach_prob", "mean"))
        .sort_values(["risk", "tickets"], ascending=[False, False]).head(15)
    )
    fig3 = px.scatter(by_cat, x="risk", y="category", size="tickets", color="dept",
                      title="Top categories by risk and volume", labels={"risk":"Mean predicted breach","tickets":"Ticket count"})
    c3.plotly_chart(fig3, use_container_width=True)

    fig4 = px.histogram(fdf, x="pred_sla_breach_prob", nbins=30, title="Risk distribution",
                        histnorm="percent", labels={"pred_sla_breach_prob":"Predicted breach probability"})
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
    fig5 = px.line(daily, x="day", y="tickets", title="Daily ticket volume", labels={"day":"Date"})
    t1.plotly_chart(fig5, use_container_width=True)

    fig6 = px.line(daily, x="day", y=["risk","breaches"], title="Predicted risk vs observed breach rate",
                   labels={"day":"Date","value":"Rate","variable":"Series"})
    t2.plotly_chart(fig6, use_container_width=True)

    fdf["week"] = fdf["created_at"].dt.to_period("W").dt.start_time
    weekly_dept = (
        fdf.groupby(["week", "dept"], as_index=False)["ticket_id"]
        .count().rename(columns={"ticket_id":"tickets"}).sort_values("week")
    )
    fig7 = px.area(weekly_dept, x="week", y="tickets", color="dept",
                   title="Weekly tickets by department", labels={"week":"Week start"})
    st.plotly_chart(fig7, use_container_width=True)

# ====== Drivers ======
with tab_drivers:
    left, right = st.columns([1, 1])

    fig8 = px.bar(fi.head(15), x="importance_mean", y="feature", orientation="h",
                  error_x="importance_std", title="Top drivers (permutation importance)",
                  labels={"importance_mean":"Mean importance"})
    left.plotly_chart(fig8, use_container_width=True)

    df_bins = fdf.copy()
    df_bins["sentiment_bin"] = pd.qcut(df_bins["citizen_sentiment"], q=10, duplicates="drop")
    df_bins["backlog_bin"]   = pd.qcut(df_bins["historical_backlog"], q=10, duplicates="drop")
    df_bins["resource_bin"]  = pd.qcut(df_bins["resource_availability"], q=10, duplicates="drop")

    feature = right.selectbox("Univariate driver view",
                              ["severity","sentiment_bin","backlog_bin","resource_bin","festival_season"], index=0)

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
    fig9 = px.bar(view, x=x_col, y="pred_sla_breach_prob", title=f"Risk by {feature}")
    right.plotly_chart(fig9, use_container_width=True)

# ====== Priority Queue (interactive ministry routing) ======
with tab_queue:
    st.subheader("Interactive Priority Queue")

    # What-if weights
    c1, c2, c3, c4, c5 = st.columns(5)
    w = {
        "risk":    c1.slider("Weight: Risk",      0.0, 1.0, 0.50, 0.01),
        "severity":c2.slider("Weight: Severity",  0.0, 1.0, 0.20, 0.01),
        "vuln":    c3.slider("Weight: Vulnerable",0.0, 1.0, 0.15, 0.01),
        "sent":    c4.slider("Weight: Sentiment", 0.0, 1.0, 0.10, 0.01),
        "fest":    c5.slider("Weight: Festival",  0.0, 1.0, 0.05, 0.01),
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
            sub = fdf[fdf["dept"] == d].sort_values(["priority_dynamic","pred_sla_breach_prob"], ascending=False)
            alloc_dept.append(sub.head(int(dept_cap[d])))
        alloc_dept = pd.concat(alloc_dept) if alloc_dept else fdf.head(0)

        alloc_min = []
        for m in min_list:
            sub = fdf[fdf["ministry_default"] == m].sort_values(["priority_dynamic","pred_sla_breach_prob"], ascending=False)
            alloc_min.append(sub.head(int(min_cap[m])))
        alloc_min = pd.concat(alloc_min) if alloc_min else fdf.head(0)

        tA, tB = st.columns(2)
        tA.caption("Dept allocation preview")
        tA.dataframe(
            alloc_dept[["ticket_id","dept","district","priority_dynamic","pred_sla_breach_prob","severity"]],
            hide_index=True, use_container_width=True,
            column_config={
                "priority_dynamic": st.column_config.NumberColumn("Priority*", format="%.1f"),
                "pred_sla_breach_prob": st.column_config.NumberColumn("P(breach)", format="%.2f"),
            }
        )
        tB.caption("Ministry allocation preview")
        tB.dataframe(
            alloc_min[["ticket_id","ministry_default","district","priority_dynamic","pred_sla_breach_prob","severity"]],
            hide_index=True, use_container_width=True,
            column_config={
                "priority_dynamic": st.column_config.NumberColumn("Priority*", format="%.1f"),
                "pred_sla_breach_prob": st.column_config.NumberColumn("P(breach)", format="%.2f"),
            }
        )

    # Editable queue with ministry routing
    editable = fdf.sort_values(["priority_dynamic","pred_sla_breach_prob"], ascending=False).copy()
    editable.insert(0, "select", False)
    editable.insert(1, "escalate", False)
    editable.insert(2, "route_dept", editable["dept"])
    editable.insert(3, "route_ministry", editable["ministry_default"])
    editable["vulnerable_population"] = editable["vulnerable_population"].astype(bool)

    dept_options = sorted(editable["dept"].unique().tolist()) + ["Cross-Dept"]

    edited = st.data_editor(
        editable[
            ["select","escalate","route_dept","route_ministry",
             "ticket_id","created_at","district","ward","dept","category",
             "severity","vulnerable_population","pred_sla_breach_prob","priority_dynamic",
             "historical_backlog","reported_via"]
        ],
        hide_index=True,
        use_container_width=True,
        column_config={
            "select": st.column_config.CheckboxColumn("Select"),
            "escalate": st.column_config.CheckboxColumn("Escalate"),
            "route_dept": st.column_config.SelectboxColumn("Route Dept", options=dept_options),
            "route_ministry": st.column_config.SelectboxColumn("Route Ministry", options=MINISTRY_OPTIONS),
            "created_at": st.column_config.DatetimeColumn("Created at"),
            "pred_sla_breach_prob": st.column_config.NumberColumn("P(breach)", format="%.2f"),
            "priority_dynamic": st.column_config.NumberColumn("Priority*", format="%.1f"),
            "vulnerable_population": st.column_config.CheckboxColumn("Vulnerable HHs"),
        },
        disabled=["ticket_id","created_at","district","ward","dept","category",
                  "pred_sla_breach_prob","priority_dynamic","historical_backlog","reported_via","severity","vulnerable_population"],
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

    cA, cB, cC = st.columns([1,1,2])

    # Export selected CSV
    cA.download_button(
        "Export selected CSV",
        sel.drop(columns=["select"]).to_csv(index=False).encode("utf-8"),
        file_name="selected_queue.csv",
        mime="text/csv",
        disabled=len(sel) == 0
    )

    # Export selected JSONL
    payload = build_payload(sel)
    jsonl = "\n".join([json.dumps(p, ensure_ascii=False) for p in payload]).encode("utf-8")
    cB.download_button(
        "Export selected JSONL",
        data=jsonl,
        file_name="selected_queue.jsonl",
        mime="application/json",
        disabled=len(sel) == 0
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

