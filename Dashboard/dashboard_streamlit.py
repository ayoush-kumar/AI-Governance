# governance_prototype/dashboard_streamlit.py
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
import streamlit as st

st.set_page_config(page_title="AI-Governance Platform", layout="wide", page_icon="🏛️")

# ---------- Ministry routing map ----------
MINISTRY_MAP = {
    "Health":    "Public Health",
    "Roads":     "Public Works",
    "Water":     "Water Supply & Sanitation",
    "Education": "School Education & Sports",
    "Safety":    "Urban Development",
}
MINISTRY_OPTIONS = sorted(set(MINISTRY_MAP.values()) | {"Cross-Ministry"})

# ---------- Plain-English glossary ----------
GLOSSARY = {
    "pred_sla_breach_prob": ("Risk Score", "AI-predicted probability that this ticket will miss its service-level deadline."),
    "sla_breach": ("Actual Breach Rate", "Percentage of tickets that actually missed their deadline."),
    "priority_score": ("Priority (Baseline)", "Initial priority score from fixed formula."),
    "priority_dynamic": ("Priority", "Priority recomputed from your weight adjustments. Range 0–100."),
    "severity": ("Severity", "Impact level: 1=low, 2=medium, 3=high."),
    "citizen_sentiment": ("Citizen Sentiment", "Public sentiment: −1=very negative, +1=very positive."),
    "vulnerable_population": ("Vulnerable Households", "Ticket affects vulnerable households."),
    "festival_season": ("Festival Period", "Issue occurred during major festival season."),
    "historical_backlog": ("Backlog", "Number of pending tickets in this area."),
    "resource_availability": ("Staff Available", "Relative staffing/resources available (0-1 scale)."),
}

def label(term: str) -> str:
    return GLOSSARY.get(term, (term.replace("_", " ").title(), ""))[0]

def help_text(term: str) -> str:
    return GLOSSARY.get(term, ("", ""))[1]

# ---------- Data loading ----------
@st.cache_data
def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["pred_sla_breach_prob"] = pd.to_numeric(df["pred_sla_breach_prob"], errors="coerce").clip(0, 1).fillna(0.0)
    df["priority_score"] = pd.to_numeric(df["priority_score"], errors="coerce").clip(0, 100).fillna(0.0)
    return df

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

def color_risk(val):
    """Color code risk values: green=low, yellow=medium, red=high"""
    if val < 0.2:
        return 'background-color: #90EE90'
    elif val < 0.3:
        return 'background-color: #FFD700'
    else:
        return 'background-color: #FFB6C1'

DATA_PATH = "governance_prototype/predictions.csv"
df = load_predictions(DATA_PATH)

# ---------- Sidebar filters ----------
st.sidebar.title("Filters & Scenarios")
scenario = st.sidebar.radio(
    "Quick Scenarios:",
    ["Custom Filters", "Monsoon Crisis", "Festival Period", "Vulnerable Households"],
    index=0,
    help="Pre-configured filters for common use cases"
)

# Define scenario-specific filter options
if scenario == "Monsoon Crisis":
    available_districts = ["Mumbai City", "Mumbai Suburban", "Thane"]
    available_depts = ["Roads", "Water"]
    st.sidebar.info(" Monsoon Crisis scenario active. Filters below refine this selection.")
elif scenario == "Festival Period":
    # For festival, allow all districts but highlight in message
    available_districts = sorted(df["district"].dropna().unique())
    available_depts = sorted(df["dept"].dropna().unique())
    st.sidebar.info(" Festival Period scenario active. Filters below refine festival-season tickets.")
elif scenario == "Vulnerable Households":
    # For vulnerable households, allow all
    available_districts = sorted(df["district"].dropna().unique())
    available_depts = sorted(df["dept"].dropna().unique())
    st.sidebar.info(" Vulnerable Households scenario active. Filters below refine this selection.")
else:
    # Custom Filters - show everything
    available_districts = sorted(df["district"].dropna().unique())
    available_depts = sorted(df["dept"].dropna().unique())
    st.sidebar.info("⚙️ Custom filter mode. All filters are independent.")

st.sidebar.markdown("---")
st.sidebar.subheader("Date Range")
min_date, max_date = df["created_at"].min().date(), df["created_at"].max().date()
date_range = st.sidebar.slider("Select date range", min_value=min_date, max_value=max_date, value=(min_date, max_date))

st.sidebar.subheader("Geographic Filters")
districts = st.sidebar.multiselect(
    "Districts", 
    available_districts,
    placeholder="All" if scenario == "Custom Filters" else f"All ({len(available_districts)} available)"
)
depts = st.sidebar.multiselect(
    "Departments", 
    available_depts,
    placeholder="All" if scenario == "Custom Filters" else f"All ({len(available_depts)} available)"
)
wards = st.sidebar.multiselect(
    "Wards", 
    sorted(df["ward"].dropna().unique()),
    placeholder="All wards"
)

st.sidebar.subheader("Risk & Priority")
severity_min = st.sidebar.select_slider(label("severity"), options=[1, 2, 3], value=1, help=help_text("severity"))
prob_min, prob_max = st.sidebar.slider(label("pred_sla_breach_prob") + " Range", 0.0, 1.0, (0.0, 1.0), 0.01, help=help_text("pred_sla_breach_prob"))
prio_min, prio_max = st.sidebar.slider(label("priority_score") + " Range", 0.0, 100.0, (0.0, 100.0), 1.0, help=help_text("priority_score"))

st.sidebar.subheader("Search")
search_text = st.sidebar.text_input("Search tickets", "", placeholder="Ticket ID, category, or ward")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Risk Level Reference:**  
- **Low:** <20%  
- **Medium:** 20-30%  
- **High:** >30%
""")

# ---------- Apply scenario presets ----------
mask = (
    (df["created_at"].dt.date >= date_range[0])
    & (df["created_at"].dt.date <= date_range[1])
    & (df["severity"] >= severity_min)
    & (df["pred_sla_breach_prob"].between(prob_min, prob_max))
    & (df["priority_score"].between(prio_min, prio_max))
)

if scenario == "Monsoon Crisis":
    monsoon_districts = ["Mumbai City", "Mumbai Suburban", "Thane"]
    monsoon_depts = ["Roads", "Water"]
    mask &= df["district"].isin(monsoon_districts)
    mask &= df["dept"].isin(monsoon_depts)
    mask &= df["severity"] >= 2

elif scenario == "Festival Period":
    mask &= (df["festival_season"] == 1)

elif scenario == "Vulnerable Households":
    mask &= (df["vulnerable_population"] == 1)

else:
    # Custom filters - only apply if user selected specific values
    if districts: mask &= df["district"].isin(districts)
    if depts: mask &= df["dept"].isin(depts)
    if wards: mask &= df["ward"].isin(wards)

# Always apply user-selected refinements on top of scenario
if scenario != "Custom Filters":
    if districts: mask &= df["district"].isin(districts)
    if wards: mask &= df["ward"].isin(wards)

if search_text.strip():
    s = search_text.strip().lower()
    mask &= (
        df["ticket_id"].fillna("").str.lower().str.contains(s, na=False)
        | df["category"].fillna("").str.lower().str.contains(s, na=False)
        | df["ward"].fillna("").str.lower().str.contains(s, na=False)
    )

fdf = df[mask].copy()

# ---------- SLA Urgency Calculation ----------
SLA_DEADLINE_HOURS = 72  # 3 days standard SLA
now = datetime.now()
fdf["hours_since_creation"] = (now - pd.to_datetime(fdf["created_at"])).dt.total_seconds() / 3600
fdf["hours_until_breach"] = SLA_DEADLINE_HOURS - fdf["hours_since_creation"]

# ---------- Header ----------
st.title("AI-Powered Governance Platform: Ministry Control Center")
st.markdown("**Super Admin Dashboard** | Smart ticket allocation and ministry-level oversight")
st.markdown("---")

# ---------- KPIs ----------
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Total Tickets",
        value=f"{len(fdf):,}",
        help="Total number of tickets in filtered dataset"
    )

with col2:
    mean_risk = fdf["pred_sla_breach_prob"].mean() if len(fdf) else 0
    risk_status = "HIGH" if mean_risk > 0.30 else "MEDIUM" if mean_risk > 0.20 else "LOW"
    st.metric(
        label="Risk Score",
        value=f"{mean_risk:.1%}",
        delta=risk_status,
        delta_color="inverse" if mean_risk > 0.30 else "off",
        help=help_text("pred_sla_breach_prob")
    )

with col3:
    breach_rate = fdf["sla_breach"].mean() if len(fdf) else 0
    breach_status = "HIGH" if breach_rate > 0.30 else "MEDIUM" if breach_rate > 0.20 else "LOW"
    st.metric(
        label="Actual Breach Rate",
        value=f"{breach_rate:.1%}",
        delta=breach_status,
        delta_color="inverse" if breach_rate > 0.30 else "off",
        help=help_text("sla_breach")
    )

with col4:
    avg_priority = fdf["priority_score"].mean() if len(fdf) else 0
    st.metric(
        label="Avg Priority (Baseline)",
        value=f"{avg_priority:.1f}",
        help=help_text("priority_score")
    )

with col5:
    avg_backlog = fdf["historical_backlog"].mean() if len(fdf) else 0
    st.metric(
        label="Avg Backlog",
        value=f"{avg_backlog:.0f}",
        help=help_text("historical_backlog")
    )

if fdf.empty:
    st.warning("No records match the current filters. Please adjust your selection.")
    st.stop()

st.markdown("---")

# ---------- Executive Summary ----------
# col_left, col_right = st.columns([1, 1])

# with col_left:
#     st.subheader("Key Findings")
#     high_risk = fdf[fdf["pred_sla_breach_prob"] > 0.6]
#     vulnerable_high_risk = high_risk[high_risk["vulnerable_population"] == 1]
#     top_dept = fdf.groupby("dept")["pred_sla_breach_prob"].mean().idxmax() if len(fdf) > 0 else "N/A"
#     worst_district = fdf.groupby("district")["sla_breach"].mean().idxmax() if len(fdf) > 0 else "N/A"

#     st.markdown(f"""
#     - **{len(high_risk):,} tickets** classified as high-risk (>60% breach probability)
#     - **{len(vulnerable_high_risk)} vulnerable households** require urgent attention
#     - **Highest risk department:** {top_dept}
#     - **Underperforming district:** {worst_district}
#     """)

# with col_right:
#     st.subheader("Recommended Actions")
#     action_count = min(50, len(high_risk))
#     top_category = fdf["category"].mode()[0] if len(fdf) > 0 else "N/A"
#     top_district_risk = fdf.groupby("district")["pred_sla_breach_prob"].mean().idxmax() if len(fdf) > 0 else "N/A"

#     st.markdown(f"""
#     - Prioritize **{action_count}** high-risk tickets
#     - Allocate resources to **{top_district_risk}**
#     - Address **{top_category}** category issues
#     """)
# ---------- Executive Summary ----------
col_left, col_right = st.columns([1, 1])


with col_left:
    st.subheader("Key Findings")
    
    # Multi-tier risk classification
    critical_tickets = fdf[
        (fdf["pred_sla_breach_prob"] > 0.80) & 
        (fdf["hours_until_breach"] < 24) &
        (fdf["hours_until_breach"] > 0)
    ]
    
    dire_tickets = fdf[
        (fdf["pred_sla_breach_prob"] > 0.80) &
        (fdf["hours_until_breach"].between(24, 72))
    ]
    
    high_risk = fdf[(fdf["pred_sla_breach_prob"] > 0.60) & (fdf["pred_sla_breach_prob"] <= 0.80)]
    
    vulnerable_critical = critical_tickets[critical_tickets["vulnerable_population"] == 1]
    top_dept = fdf.groupby("dept")["pred_sla_breach_prob"].mean().idxmax() if len(fdf) > 0 else "N/A"
    worst_district = fdf.groupby("district")["sla_breach"].mean().idxmax() if len(fdf) > 0 else "N/A"

    st.markdown(f"""
    - **{len(critical_tickets):,} critical tickets** (<24hrs to SLA breach + >80% risk)
    - **{len(vulnerable_critical)} vulnerable households** in critical tier
    - **{len(dire_tickets):,} dire tickets** (24-72hrs window + >80% risk)
    - **Highest risk department:** {top_dept}
    - **Underperforming district:** {worst_district}
    """)


with col_right:
    st.subheader("Recommended Actions")
    
    top_category = fdf["category"].mode()[0] if len(fdf) > 0 else "N/A"
    top_district_risk = fdf.groupby("district")["pred_sla_breach_prob"].mean().idxmax() if len(fdf) > 0 else "N/A"
    
    st.markdown(f"""
    - **CRITICAL:** Resolve {len(critical_tickets)} tickets immediately (<24hrs)
    - **DIRE:** Prioritize {len(dire_tickets)} tickets within 48 hours
    - **ELEVATED:** Proactively address {len(high_risk)} tickets this week
    - **Focus areas:** {top_district_risk} district, {top_category} category
    """)
    
    # Smart multi-tier capacity alert
    if len(critical_tickets) > 50:
        st.error(f"CRISIS MODE: {len(critical_tickets)} critical tickets exceed emergency capacity. Immediate action required.")
    elif len(critical_tickets) > 20:
        st.error(f"HIGH ALERT: {len(critical_tickets)} critical tickets require immediate resolution.")
    elif len(critical_tickets) > 0:
        st.warning(f"URGENT: {len(critical_tickets)} tickets breach SLA within 24 hours.")
    elif len(dire_tickets) > 50:
        st.warning(f"ELEVATED ALERT: {len(dire_tickets)} dire tickets need prioritization within 48 hours.")
    elif len(dire_tickets) > 0:
        st.info(f"ATTENTION: {len(dire_tickets)} dire tickets require action within 48-72 hours.")
    elif len(high_risk) > 30:
        st.info(f"PROACTIVE MODE: {len(high_risk)} elevated-risk tickets need weekly planning.")
    elif len(high_risk) > 0:
        st.success(f" STABLE: {len(high_risk)} elevated tickets identified for proactive resolution.")
    else:
        st.success(" OPTIMAL: No urgent tickets. System operating at baseline.")


st.markdown("---")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Distribution Analysis",
    "Trend Analysis",
    "Impact Assessment",
    "Action Queue",
    "Monitoring & Oversight"
])

# ====== Tab 1: Distribution Analysis ======
with tab1:
    st.subheader("Risk Distribution by Region and Department")

    # Check if specific wards are selected
    show_ward_level = len(wards) > 0

    if show_ward_level:
        # Ward-level view when wards are filtered
        st.markdown("#### Ward-Level Breakdown")

        risk_by_ward = (
            fdf.groupby("ward", as_index=False)["pred_sla_breach_prob"]
            .mean().sort_values("pred_sla_breach_prob", ascending=False)
        )

        fig_ward = px.bar(
            risk_by_ward,
            x="ward",
            y="pred_sla_breach_prob",
            title=f"Risk by Ward ({len(wards)} ward(s) selected)",
            labels={"pred_sla_breach_prob": "Mean Risk Score", "ward": "Ward"},
            color="pred_sla_breach_prob",
            color_continuous_scale=["green", "yellow", "red"]
        )
        fig_ward.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_ward, use_container_width=True)

        st.markdown("---")
        st.markdown("#### District and Department Context")

    # Always show district and department breakdown
    c1, c2 = st.columns(2)

    with c1:
        risk_by_dist = (
            fdf.groupby("district", as_index=False)["pred_sla_breach_prob"]
            .mean().sort_values("pred_sla_breach_prob", ascending=False)
        )
        max_district = risk_by_dist.iloc[0]["district"] if len(risk_by_dist) > 0 else "N/A"
        max_risk = risk_by_dist.iloc[0]["pred_sla_breach_prob"] if len(risk_by_dist) > 0 else 0

        fig1 = px.bar(
            risk_by_dist,
            x="district",
            y="pred_sla_breach_prob",
            title=f"Risk by District (Highest: {max_district} at {max_risk:.1%})",
            labels={"pred_sla_breach_prob": "Mean Risk Score", "district": "District"},
            color="pred_sla_breach_prob",
            color_continuous_scale=["green", "yellow", "red"]
        )
        fig1.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        risk_by_dept = (
            fdf.groupby("dept", as_index=False)["pred_sla_breach_prob"]
            .mean().sort_values("pred_sla_breach_prob", ascending=False)
        )
        max_dept = risk_by_dept.iloc[0]["dept"] if len(risk_by_dept) > 0 else "N/A"

        fig2 = px.bar(
            risk_by_dept,
            x="pred_sla_breach_prob",
            y="dept",
            orientation="h",
            title=f"Risk by Department (Highest: {max_dept})",
            labels={"pred_sla_breach_prob": "Mean Risk Score", "dept": "Department"},
            color="pred_sla_breach_prob",
            color_continuous_scale=["green", "yellow", "red"]
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### Category-Level Analysis")

    # Adapt category analysis based on ward selection
    if show_ward_level:
        by_cat = (
            fdf.groupby(["ward", "category"], as_index=False)
            .agg(tickets=("ticket_id", "count"), risk=("pred_sla_breach_prob", "mean"))
            .sort_values(["risk", "tickets"], ascending=[False, False]).head(20)
        )
        fig3 = px.scatter(
            by_cat,
            x="risk",
            y="category",
            size="tickets",
            color="ward",
            title="High-Risk Categories by Ward (bubble size = ticket volume)",
            labels={"risk": "Mean Risk Score", "tickets": "Ticket Count", "category": "Category", "ward": "Ward"},
        )
    else:
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
            title="High-Risk Categories by Department (bubble size = ticket volume)",
            labels={"risk": "Mean Risk Score", "tickets": "Ticket Count", "category": "Category", "dept": "Department"},
        )

    fig3.update_layout(height=500)
    st.plotly_chart(fig3, use_container_width=True)

# ====== Tab 2: Trend Analysis ======
with tab2:
    st.subheader("Temporal Patterns and Trends")

    fdf["day"] = fdf["created_at"].dt.date
    daily = (
        fdf.groupby("day", as_index=False)
        .agg(tickets=("ticket_id", "count"), risk=("pred_sla_breach_prob", "mean"), breaches=("sla_breach", "mean"))
        .sort_values("day")
    )

    t1, t2 = st.columns(2)

    with t1:
        peak_day = daily.loc[daily["tickets"].idxmax(), "day"] if len(daily) > 0 else "N/A"
        peak_count = daily["tickets"].max() if len(daily) > 0 else 0

        fig5 = px.line(
            daily,
            x="day",
            y="tickets",
            title=f"Daily Ticket Volume (Peak: {peak_day}, {peak_count} tickets)",
            labels={"day": "Date", "tickets": "Tickets"}
        )
        fig5.update_traces(line_color="#1f77b4")
        st.plotly_chart(fig5, use_container_width=True)

    with t2:
        fig6 = px.line(
            daily,
            x="day",
            y=["risk", "breaches"],
            title="Model Accuracy: Predicted Risk vs Actual Breach Rate",
            labels={"day": "Date", "value": "Rate", "variable": "Metric"},
        )
        fig6.update_traces(name="AI Predicted Risk", selector=dict(name="risk"))
        fig6.update_traces(name="Actual Breaches", selector=dict(name="breaches"))
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("### Weekly Department Trends")
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
        title="Weekly Ticket Volume by Department",
        labels={"week": "Week Starting", "tickets": "Tickets", "dept": "Department"},
    )
    st.plotly_chart(fig7, use_container_width=True)

# ====== Tab 3: Impact Assessment ======
with tab3:
    st.subheader("Resource Planning and Impact Calculator")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Resource Allocation")
        extra_staff = st.slider("Additional field staff (%)", 0, 50, 0, 5, help="Simulate impact of hiring additional staff")
        focus_dept = st.selectbox("Priority department", sorted(fdf["dept"].unique()))

        st.markdown("---")
        st.markdown("#### Current Metrics")
        current_tickets = len(fdf)
        current_breach = fdf["sla_breach"].mean()
        high_risk_count = len(fdf[fdf["pred_sla_breach_prob"] > 0.6])

        st.metric("Active Tickets", f"{current_tickets:,}")
        st.metric("Current Breach Rate", f"{current_breach:.1%}")
        st.metric("High-Risk Tickets", f"{high_risk_count:,}")

    with col2:
        st.markdown("#### Projected Impact")

        # Impact model: 10% staff increase ≈ 3% breach reduction
        projected_breach = current_breach * (1 - (extra_staff * 0.003))
        prevented_breaches = int((current_breach - projected_breach) * current_tickets)

        # Cost calculation: ₹3000 penalty per breach
        penalty_per_breach = 3000
        current_cost = int(current_breach * current_tickets * penalty_per_breach)
        projected_cost = int(projected_breach * current_tickets * penalty_per_breach)
        savings = current_cost - projected_cost

        st.metric(
            "Projected Breach Rate",
            f"{projected_breach:.1%}",
            delta=f"{(projected_breach - current_breach):.1%}",
            delta_color="inverse"
        )
        st.metric(
            "Breaches Prevented",
            f"{prevented_breaches:,}",
            help=f"Estimated reduction in SLA breaches"
        )
        st.metric(
            "Estimated Cost Savings",
            f"₹{savings:,}",
            delta=f"₹{savings:,} penalty reduction",
            help="Based on ₹3,000 penalty per breach"
        )

    st.markdown("---")
    st.markdown("#### Department Performance Comparison")

    dept_stats = fdf.groupby("dept").agg({
        "ticket_id": "count",
        "sla_breach": "mean",
        "pred_sla_breach_prob": "mean",
        "resolution_hours": "mean"
    }).round(3)
    dept_stats.columns = ["Total Tickets", "Actual Breach Rate", "AI Risk Score", "Avg Resolution (hrs)"]
    dept_stats = dept_stats.sort_values("AI Risk Score", ascending=False)

    styled_df = dept_stats.style.applymap(color_risk, subset=["AI Risk Score", "Actual Breach Rate"])
    st.dataframe(styled_df, use_container_width=True)

# ====== Tab 4: Action Queue ======
with tab4:
    st.subheader("Priority Queue and Ticket Assignment")

    # Priority weight adjustment
    with st.expander("Adjust Priority Weights", expanded=False):
        st.markdown("Customize priority scoring based on organizational policy")
        c1, c2, c3, c4, c5 = st.columns(5)
        w = {
            "risk":    c1.slider("Risk Weight", 0.0, 1.0, 0.50, 0.01, help=help_text("pred_sla_breach_prob")),
            "severity":c2.slider("Severity Weight", 0.0, 1.0, 0.20, 0.01, help=help_text("severity")),
            "vuln":    c3.slider("Vulnerability Weight", 0.0, 1.0, 0.15, 0.01, help=help_text("vulnerable_population")),
            "sent":    c4.slider("Sentiment Weight", 0.0, 1.0, 0.10, 0.01, help=help_text("citizen_sentiment")),
            "fest":    c5.slider("Festival Weight", 0.0, 1.0, 0.05, 0.01, help=help_text("festival_season")),
        }

    fdf = fdf.copy()
    fdf["priority_dynamic"] = compute_dynamic_priority(fdf, w)
    fdf["ministry_default"] = fdf["dept"].map(MINISTRY_MAP).fillna("Cross-Ministry")

    # Editable queue
    editable = fdf.sort_values(["priority_dynamic", "pred_sla_breach_prob"], ascending=False).copy()
    editable.insert(0, "select", False)
    editable.insert(1, "escalate", False)
    editable.insert(2, "route_dept", editable["dept"])
    editable.insert(3, "route_ministry", editable["ministry_default"])
    editable["vulnerable_population"] = editable["vulnerable_population"].astype(bool)

    dept_options = sorted(editable["dept"].unique().tolist()) + ["Cross-Department"]

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
            ]
        ].head(100),
        hide_index=True,
        use_container_width=True,
        column_config={
            "select": st.column_config.CheckboxColumn("Select"),
            "escalate": st.column_config.CheckboxColumn("Escalate"),
            "route_dept": st.column_config.SelectboxColumn("Route to Department", options=dept_options),
            "route_ministry": st.column_config.SelectboxColumn("Route to Ministry", options=MINISTRY_OPTIONS),
            "created_at": st.column_config.DatetimeColumn("Created Date"),
            "dept": "Department",
            "district": "District",
            "ward": "Ward",
            "category": "Category",
            "pred_sla_breach_prob": st.column_config.NumberColumn("Risk Score", format="%.0%%", help=help_text("pred_sla_breach_prob")),
            "priority_dynamic": st.column_config.NumberColumn("Priority Score", format="%.1f", help=help_text("priority_dynamic")),
            "severity": st.column_config.NumberColumn("Severity", help=help_text("severity")),
            "vulnerable_population": st.column_config.CheckboxColumn("Vulnerable HH", help=help_text("vulnerable_population")),
            "historical_backlog": st.column_config.NumberColumn("Backlog", help=help_text("historical_backlog")),
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
            "severity",
            "vulnerable_population",
        ],
        key="queue_editor",
    )

    # Selection actions
    sel = edited[edited["select"]]
    st.caption(f"Selected: {len(sel)} ticket(s)")

    def build_payload(rows: pd.DataFrame) -> list[dict]:
        from datetime import datetime
        return [
            {
                "ticket_id": r["ticket_id"],
                "assignment": {
                    "ministry": r["route_ministry"],
                    "department": r["route_dept"],
                    "escalated": bool(r["escalate"]),
                },
                "status": "In Progress",
                "assigned_at": datetime.now().isoformat(),
                "priority_score": float(r["priority_dynamic"]),
                "risk_score": float(r["pred_sla_breach_prob"]),
                "metadata": {
                    "district": r["district"],
                    "ward": r["ward"],
                    "category": r["category"],
                    "severity": int(r["severity"]),
                    "vulnerable_household": bool(r["vulnerable_population"]),
                },
            }
            for _, r in rows.iterrows()
        ]

    if "assignments" not in st.session_state:
        st.session_state["assignments"] = []

    col_a, col_b, col_c = st.columns([1, 1, 2])

    col_a.download_button(
        "Export Selection (CSV)",
        sel.drop(columns=["select"]).to_csv(index=False).encode("utf-8"),
        file_name="priority_tickets.csv",
        mime="text/csv",
        disabled=len(sel) == 0,
        help="Download selected tickets as CSV file"
    )

    if col_b.button("Assign Selected Tickets", disabled=len(sel) == 0, type="primary", use_container_width=True):
        payload = build_payload(sel)
        st.session_state["assignments"].extend(payload)
        st.success(f"Successfully assigned {len(sel)} ticket(s) to specified departments/ministries")
        st.info("Navigate to 'Monitoring & Oversight' tab to track assigned tickets")

    if col_c.button("Clear Assignment History", disabled=len(st.session_state["assignments"]) == 0):
        st.session_state["assignments"] = []
        st.info("Assignment history cleared")

    # Assignment log
    if st.session_state["assignments"]:
        st.markdown("---")
        st.markdown(f"#### Assignment Log ({len(st.session_state['assignments'])} total assignments)")
        recent = pd.DataFrame(st.session_state["assignments"]).tail(20).iloc[::-1]
        st.dataframe(recent, hide_index=True, use_container_width=True, height=300)

# ====== Tab 5: Monitoring & Oversight ======
with tab5:
    st.subheader("Assignment Monitoring and Ministry Performance")

    # Get assigned tickets from session state
    if "assignments" in st.session_state and len(st.session_state["assignments"]) > 0:
        assigned_df = pd.DataFrame(st.session_state["assignments"])

        # Parse assignment data
        assigned_df["ministry"] = assigned_df["assignment"].apply(lambda x: x["ministry"] if isinstance(x, dict) else "Unknown")
        assigned_df["department"] = assigned_df["assignment"].apply(lambda x: x["department"] if isinstance(x, dict) else "Unknown")
        assigned_df["assigned_at_dt"] = pd.to_datetime(assigned_df["assigned_at"])

        # Calculate time since assignment
        now = datetime.now()
        assigned_df["hours_since_assignment"] = assigned_df["assigned_at_dt"].apply(
            lambda x: (now - x).total_seconds() / 3600
        )

        # Overview metrics
        st.markdown("### Assignment Overview")
        m1, m2, m3, m4 = st.columns(4)

        total_assigned = len(assigned_df)
        in_progress = len(assigned_df[assigned_df["status"] == "In Progress"])
        pending = 0
        resolved = 0

        m1.metric("Total Assigned", f"{total_assigned}")
        m2.metric("In Progress", f"{in_progress}", delta=f"{in_progress/total_assigned*100:.0f}%" if total_assigned > 0 else "0%")
        m3.metric("Pending Action", f"{pending}")
        m4.metric("Resolved", f"{resolved}")

        st.markdown("---")

        # Ministry Performance
        st.markdown("### Ministry Performance Dashboard")

        ministry_perf = assigned_df.groupby("ministry").agg({
            "ticket_id": "count",
            "hours_since_assignment": "mean",
            "risk_score": "mean"
        }).reset_index()
        ministry_perf.columns = ["Ministry", "Assigned Tickets", "Avg Hours Since Assignment", "Avg Risk Score"]
        ministry_perf = ministry_perf.sort_values("Assigned Tickets", ascending=False)

        # Add status indicator
        ministry_perf["Status"] = ministry_perf["Avg Hours Since Assignment"].apply(
            lambda x: "🟢 On Track" if x < 24 else "🟡 Monitor" if x < 48 else "🔴 Attention Needed"
        )

        st.dataframe(
            ministry_perf.style.format({
                "Assigned Tickets": "{:,.0f}",
                "Avg Hours Since Assignment": "{:.1f}",
                "Avg Risk Score": "{:.1%}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Tickets by Ministry")
            fig_ministry = px.bar(
                ministry_perf,
                x="Ministry",
                y="Assigned Tickets",
                color="Avg Risk Score",
                title="Assignment Distribution",
                color_continuous_scale=["green", "yellow", "red"]
            )
            fig_ministry.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_ministry, use_container_width=True)

        with col2:
            st.markdown("#### Time Since Assignment")
            fig_time = px.bar(
                ministry_perf,
                x="Ministry",
                y="Avg Hours Since Assignment",
                title="Average Response Time",
                color="Avg Hours Since Assignment",
                color_continuous_scale=["green", "yellow", "red"]
            )
            fig_time.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_time, use_container_width=True)

        st.markdown("---")

        # Alerts for stuck tickets
        st.markdown("### Active Alerts")
        stuck_threshold = 72
        stuck_tickets = assigned_df[assigned_df["hours_since_assignment"] > stuck_threshold]

        if len(stuck_tickets) > 0:
            st.warning(f"⚠️ **{len(stuck_tickets)} tickets** assigned >72 hours ago with no resolution")

            stuck_by_ministry = stuck_tickets.groupby("ministry")["ticket_id"].count().sort_values(ascending=False)
            for ministry, count in stuck_by_ministry.items():
                st.markdown(f"- **{ministry}**: {count} ticket(s)")
        else:
            st.success("No stuck tickets. All assignments within expected timeline.")

        st.markdown("---")

        # Recent Assignments Detail
        st.markdown("### Recent Assignments")
        recent_assignments = assigned_df.sort_values("assigned_at_dt", ascending=False).head(20)

        display_df = recent_assignments[[
            "ticket_id", "ministry", "department", "status", 
            "risk_score", "priority_score", "hours_since_assignment"
        ]].copy()
        display_df.columns = [
            "Ticket ID", "Ministry", "Department", "Status", 
            "Risk Score", "Priority", "Hours Since Assignment"
        ]

        st.dataframe(
            display_df.style.format({
                "Risk Score": "{:.0%}",
                "Priority": "{:.1f}",
                "Hours Since Assignment": "{:.1f}"
            }),
            use_container_width=True,
            hide_index=True,
            height=400
        )

    else:
        st.info("📊 No assignments yet. Go to 'Action Queue' tab to assign tickets to ministries.")
        st.markdown("""
        **How to use this tab:**
        1. Navigate to the **Action Queue** tab
        2. Select tickets to assign
        3. Click **Assign Selected Tickets**
        4. Return here to monitor their progress

        This tab will show:
        - Assignment status overview
        - Ministry performance metrics
        - Time-since-assignment tracking
        - Stuck ticket alerts
        """)

st.markdown("---")
st.caption("AI-Powered Governance Platform • Advanced Analytics for Public Service Management")