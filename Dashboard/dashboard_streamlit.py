# governance_prototype/dashboard_streamlit.py
import json
import uuid
import time
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Google Cloud Libraries
from google.cloud import bigquery
from google.cloud import firestore
import vertexai
from vertexai.generative_models import GenerativeModel

# Initialize Cloud Clients
PROJECT_ID = "hackathon-pipeline"
REGION = "asia-south1"
db = firestore.Client(project=PROJECT_ID)

st.set_page_config(page_title="AI-Governance Platform", layout="wide", page_icon="🏛️")

# ---------- Impact Assessment Configuration ----------
# Service-specific SLA and penalties (based on Delhi e-SLA model)
SLA_PENALTY_MODEL = {
    "Health": {"sla_days": 7, "penalty_per_day": 10, "max_penalty": 200},
    "Roads": {"sla_days": 15, "penalty_per_day": 10, "max_penalty": 200},
    "Water": {"sla_days": 10, "penalty_per_day": 10, "max_penalty": 200},
    "Education": {"sla_days": 30, "penalty_per_day": 10, "max_penalty": 200},
    "Safety": {"sla_days": 7, "penalty_per_day": 10, "max_penalty": 200}
}

# Staff and operational costs (Maharashtra government scale)
AVG_MONTHLY_SALARY_FIELD_STAFF = 20000  # Rs 20,000/month (realistic for field staff)
TRAINING_COST_PER_OFFICER = 3000        # One-time Rs 3,000 (amortized over months)
OVERHEAD_MULTIPLIER = 1.15              # 15% overhead

# Department baseline staffing
BASELINE_STAFF = {
    "Health": 50,
    "Roads": 60,
    "Water": 40,
    "Education": 45,
    "Safety": 35
}

# Benchmark data from research
BENCHMARK_DATA = {
    "Maharashtra (Current)": {"applications_per_year": 176000000, "breach_rate": 0.35, "avg_resolution_days": 12},
    "Delhi (e-SLA)": {"applications_per_year": 45000000, "breach_rate": 0.28, "avg_resolution_days": 9},
    "Karnataka (Seva Sindhu)": {"applications_per_year": 80000000, "breach_rate": 0.32, "avg_resolution_days": 11},
    "National Average (UMANG)": {"applications_per_year": 383000000, "breach_rate": 0.40, "avg_resolution_days": 15}
}

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
    # NEW FEATURES FROM UPDATED PIPELINE
    "escalated": ("Escalated", "Ticket escalated to higher authority (1=Yes, 0=No)."),
    "escalation_level": ("Escalation Level", "0=None, 1=Supervisor, 2=District Officer, 3=Ministry."),
    "escalation_days": ("Escalation Time", "Days taken to resolve after escalation."),
    "created_hour": ("Created Hour", "Hour of day when ticket was created (0-23)."),
    "is_business_hours": ("Business Hours", "Ticket created during business hours (9 AM - 5 PM)."),
    "is_weekend": ("Weekend", "Ticket created on weekend (Saturday/Sunday)."),
    "sla_deadline_hours": ("SLA Deadline", "Department-specific SLA deadline in hours."),
}

# ---------- Impact Assessment Helper Functions ----------

def calculate_penalty(dept, delay_days):
    """Calculate penalty based on department-specific SLA"""
    model = SLA_PENALTY_MODEL.get(dept, {"penalty_per_day": 10, "max_penalty": 200})
    return min(delay_days * model["penalty_per_day"], model["max_penalty"])


def calculate_breach_reduction(current_breach_rate, staff_increase_pct, current_tickets, scaling_factor=1.0):
    """
    Evidence-based breach reduction model with diminishing returns and overcrowding penalties
    
    Research basis:
    - 0-25%: High efficiency (0.9x multiplier per 1% staff)
    - 25-40%: Optimal zone (0.6x multiplier - diminishing returns)
    - 40-50%: Overcrowding (0.2x multiplier - coordination overhead)
    - >50%: Negative returns (coordination breakdown)
    """
    # Adjust reduction expectations for small datasets
    effective_reduction_multiplier = min(1.0, 0.5 + (scaling_factor * 0.5))
    
    # Four-zone model with realistic diminishing returns
    if staff_increase_pct <= 25:
        # Zone 1: High efficiency
        breach_reduction_pct = staff_increase_pct * 0.9 * effective_reduction_multiplier
        
    elif staff_increase_pct <= 40:
        # Zone 2: Optimal (diminishing returns kick in)
        base_reduction = 25 * 0.9 * effective_reduction_multiplier
        extra_reduction = (staff_increase_pct - 25) * 0.6 * effective_reduction_multiplier
        breach_reduction_pct = base_reduction + extra_reduction
        
    elif staff_increase_pct <= 50:
        # Zone 3: Overcrowding (minimal gains)
        base_reduction = 25 * 0.9 + 15 * 0.6
        base_reduction *= effective_reduction_multiplier
        extra_reduction = (staff_increase_pct - 40) * 0.2 * effective_reduction_multiplier
        breach_reduction_pct = base_reduction + extra_reduction
        
    else:
        # Zone 4: Coordination breakdown (negative returns)
        base_reduction = 25 * 0.9 + 15 * 0.6 + 10 * 0.2
        base_reduction *= effective_reduction_multiplier
        # Beyond 50%, each additional 10% reduces effectiveness by 1%
        coordination_penalty = (staff_increase_pct - 50) * 0.1
        breach_reduction_pct = max(0, base_reduction - coordination_penalty)
    
    # Training period inefficiency (10%)
    breach_reduction_pct *= 0.90
    
    # Apply realistic floor: can't reduce breach rate below 5%
    new_breach_rate = max(0.05, current_breach_rate * (1 - breach_reduction_pct / 100))
    
    return new_breach_rate, breach_reduction_pct

def calculate_roi(staff_increase_pct, current_tickets, current_breach_rate, dept):
    """Calculate complete ROI including costs and savings"""
    if staff_increase_pct == 0:
        return None
    
    # Staff calculation - scale to filtered dataset
    dept_baseline_staff = BASELINE_STAFF.get(dept, 50)
    
    # Estimate: Baseline assumes ~1000 tickets/month for full department
    # Scale down costs if we're looking at a smaller subset (e.g., one district)
    baseline_ticket_assumption = 1000
    scaling_factor = min(1.0, current_tickets / baseline_ticket_assumption)
    
    # Effective staff needed for this filtered dataset
    effective_staff = max(1, int(dept_baseline_staff * scaling_factor))
    new_staff_count = max(1, int(effective_staff * (staff_increase_pct / 100)))
    
    # Minimum staff check - don't recommend hiring for very small datasets
    if current_tickets < 20:
        # Too few tickets to justify hiring analysis
        return None
    
    if new_staff_count == 0:
        new_staff_count = 1  # Minimum 1 staff member

    # Cost calculation
    monthly_salary_cost = new_staff_count * AVG_MONTHLY_SALARY_FIELD_STAFF
    training_cost = new_staff_count * TRAINING_COST_PER_OFFICER
    
    # Coordination overhead increases with staff count
    if staff_increase_pct <= 25:
        coordination_multiplier = 1.0  # Normal efficiency
    elif staff_increase_pct <= 40:
        coordination_multiplier = 1.1  # 10% coordination overhead
    elif staff_increase_pct <= 50:
        coordination_multiplier = 1.2  # 20% coordination overhead
    else:
        # Exponential overhead beyond 50%
        coordination_multiplier = 1.2 + (staff_increase_pct - 50) * 0.03  # 3% per 1% above 50%
    
    total_monthly_cost = (monthly_salary_cost + training_cost) * OVERHEAD_MULTIPLIER * coordination_multiplier
    
    # Savings calculation
    new_breach_rate, reduction_pct = calculate_breach_reduction(
        current_breach_rate, staff_increase_pct, current_tickets, scaling_factor
    )
    breaches_prevented = int((current_breach_rate - new_breach_rate) * current_tickets)
    
    # Calculate total breach cost (comprehensive government cost)
    officer_penalty = calculate_penalty(dept, delay_days=7)
    
    # Base costs per breach
    citizen_visit_cost = 3000  # 3 visits × 4 hours × Rs 250/hour
    admin_overhead = 800
    reputation_cost = 1000
    litigation_risk = 500
    
    # For smaller datasets (district/ward level), costs might be lower
    # because administrative overhead is shared across the region
    if current_tickets < 200:
        # Reduce overhead costs for small filtered datasets
        admin_overhead = int(admin_overhead * 0.6)  # Rs 480
        litigation_risk = int(litigation_risk * 0.6)  # Rs 300
    
    total_breach_cost = officer_penalty + citizen_visit_cost + admin_overhead + reputation_cost + litigation_risk
    
    monthly_savings = breaches_prevented * total_breach_cost
    
    # ROI metrics
    net_benefit = monthly_savings - total_monthly_cost
    roi_percentage = (net_benefit / total_monthly_cost * 100) if total_monthly_cost > 0 else 0
    # Payback calculation: Total upfront cost / monthly net benefit
    total_upfront_cost = training_cost  # One-time cost
    
    if net_benefit > 0:
        payback_months = total_upfront_cost / net_benefit
    else:
        payback_months = float('inf')
    
    return {
        "new_staff": new_staff_count,
        "monthly_cost": int(total_monthly_cost),
        "monthly_savings": int(monthly_savings),
        "net_benefit": int(net_benefit),
        "roi_percentage": roi_percentage,
        "payback_months": payback_months,
        "breaches_prevented": breaches_prevented,
        "new_breach_rate": new_breach_rate,
        "reduction_pct": reduction_pct
    }


def calculate_citizen_impact(breaches_prevented, current_tickets, vulnerable_count):
    """Calculate citizen-facing impact metrics"""
    # Time saved (3 follow-up visits x 2 hours each)
    citizen_hours_saved = breaches_prevented * 3 * 2
    citizen_days_saved = citizen_hours_saved / 8
    
    # Economic value (Rs 500 avg daily wage)
    economic_value = citizen_days_saved * 500
    
    # Satisfaction improvement (0.2 points per 10% breach reduction)
    satisfaction_delta = (breaches_prevented / current_tickets * 10 * 0.2) if current_tickets > 0 else 0
    
    return {
        "citizen_hours_saved": int(citizen_hours_saved),
        "citizen_days_saved": int(citizen_days_saved),
        "economic_value": int(economic_value),
        "satisfaction_delta": satisfaction_delta,
        "vulnerable_helped": vulnerable_count
    }


def label(term: str) -> str:
    return GLOSSARY.get(term, (term.replace("_", " ").title(), ""))[0]

def help_text(term: str) -> str:
    return GLOSSARY.get(term, ("", ""))[1]

# ---------- Cloud Data Functions ----------
@st.cache_data(ttl=300)
def load_predictions() -> pd.DataFrame:
    """Load data from BigQuery"""
    client = bigquery.Client(project=PROJECT_ID)
    query = f"SELECT * FROM `{PROJECT_ID}.governance_data.predictions`"
    
    df = client.query(query).to_dataframe()
    
    # Type Conversion
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["pred_sla_breach_prob"] = pd.to_numeric(df["pred_sla_breach_prob"], errors="coerce").clip(0, 1).fillna(0.0)
    df["priority_score"] = pd.to_numeric(df["priority_score"], errors="coerce").clip(0, 100).fillna(0.0)
    return df

def save_assignment_firestore(data):
    """Save assignment to Firestore"""
    doc_id = str(uuid.uuid4())
    data['saved_at'] = firestore.SERVER_TIMESTAMP
    db.collection('assignments').document(doc_id).set(data)

def get_assignments_firestore():
    """Fetch history from Firestore"""
    docs = db.collection('assignments').order_by('saved_at', direction=firestore.Query.DESCENDING).limit(50).stream()
    return [doc.to_dict() for doc in docs]

def get_gemini_response(query, context_df):
    """Real Gemini integration via Vertex AI with Enhanced Context"""
    try:
        # Initialize Vertex AI (Keep us-central1 for access to newer models)
        vertexai.init(project=PROJECT_ID, location="us-central1")
        
        # Using the model you selected
        model = GenerativeModel("gemini-2.0-flash-lite-001") 
        
        # --- BUILD RICH CONTEXT ---
        # 1. High-level aggregates
        total_tickets = len(context_df)
        avg_risk = context_df['pred_sla_breach_prob'].mean()
        
        # 2. Top Distributions (So it knows which districts/depts are active)
        district_counts = context_df['district'].value_counts().head(5).to_dict()
        dept_counts = context_df['dept'].value_counts().head(5).to_dict()
        
        # 3. Detailed sample of High Risk Tickets (The "Evidence")
        # We grab the top 25 highest risk tickets so the AI can see specific examples
        # We convert it to a CSV string so the LLM can read it easily
        high_risk_data = context_df.sort_values("pred_sla_breach_prob", ascending=False).head(25)
        # Select only relevant columns to save tokens
        sample_csv = high_risk_data[['ticket_id', 'district', 'dept', 'category', 'pred_sla_breach_prob', 'sla_breach']].to_string(index=False)

        # Construct the Prompt
        prompt = f"""
        You are an intelligent government analyst. You have access to the current live dashboard data.
        
        ### DATA SNAPSHOT:
        - **Total Tickets in View:** {total_tickets}
        - **Average Breach Risk:** {avg_risk:.1%}
        - **Top Districts (by volume):** {district_counts}
        - **Top Departments (by volume):** {dept_counts}

        ### TOP 25 CRITICAL TICKETS (Sample Data):
        {sample_csv}

        ### USER QUERY:
        "{query}"

        ### INSTRUCTIONS:
        1. Answer strictly based on the provided data above.
        2. If the user asks about a specific city (like Mumbai), look at the 'district' column in the sample data.
        3. Identify patterns (e.g., "Most critical water issues are in Mumbai City").
        4. Be concise and professional.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"

def compute_dynamic_priority(df: pd.DataFrame, w: dict) -> pd.Series:
    ws = np.array([w["risk"], w["severity"], w["vuln"], w["sent"], w["fest"]], dtype=float)
    
    # If all weights are zero, use defaults
    if ws.sum() == 0:
        ws = np.array([0.50, 0.20, 0.15, 0.10, 0.05])
        st.warning("All weights set to 0. Using default weights: Risk=50%, Severity=20%, Vulnerability=15%, Sentiment=10%, Festival=5%")
    else:
        ws = ws / ws.sum()
    
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

# DATA_PATH = "governance_prototype/predictions.csv"
# df = load_predictions(DATA_PATH)
df = load_predictions()

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

st.sidebar.subheader("Escalation Status")
escalation_filter = st.sidebar.multiselect(
    "Escalation Level",
    ["Not Escalated", "Supervisor", "District Officer", "Ministry"],
    placeholder="All escalation levels",
    help="Filter by escalation status"
)

st.sidebar.subheader("Search")
search_text = st.sidebar.text_input("Search tickets", "", placeholder="Ticket ID, category, or ward")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Risk Level Reference:**  
- **Low:** <20%  
- **Medium:** 20-30%  
- **High:** >30%

**Department SLA Deadlines:**  
- **Health:** 7 days (168 hrs)  
- **Safety:** 7 days (168 hrs)  
- **Water:** 10 days (240 hrs)  
- **Roads:** 15 days (360 hrs)  
- **Education:** 30 days (720 hrs)
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

# Apply escalation filter
if escalation_filter:
    escalation_map = {"Not Escalated": 0, "Supervisor": 1, "District Officer": 2, "Ministry": 3}
    escalation_nums = [escalation_map[e] for e in escalation_filter]
    if "escalation_level" in df.columns:
        mask &= df["escalation_level"].isin(escalation_nums)

fdf = df[mask].copy()


# ---------- SLA Urgency Calculation ----------
now = datetime.now()
fdf["hours_since_creation"] = (now - pd.to_datetime(fdf["created_at"])).dt.total_seconds() / 3600

# Calculate hours until breach using department-specific SLA from data
fdf["hours_until_breach"] = fdf["sla_deadline_hours"] - fdf["hours_since_creation"]
fdf["already_breached"] = fdf["hours_until_breach"] < 0

# Also calculate urgency using 72-hour action window for immediate prioritization
fdf["hours_until_urgent"] = 72 - fdf["hours_since_creation"]
fdf["is_urgent_72h"] = (fdf["hours_until_urgent"] > 0) & (fdf["hours_until_urgent"] < 72)


# ---------- Header ----------
st.title("AI-Powered Governance Platform: Ministry Control Center")
st.markdown("**Super Admin Dashboard** | Smart ticket allocation and ministry-level oversight")
st.markdown("---")

# ---------- KPIs ----------
col1, col2, col3, col4, col5, col6 = st.columns(6)


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

with col6:
    escalation_rate = fdf["escalated"].mean() if len(fdf) and "escalated" in fdf.columns else 0
    esc_status = "HIGH" if escalation_rate > 0.40 else "NORMAL"
    st.metric(
        label="Escalation Rate",
        value=f"{escalation_rate:.1%}",
        delta=esc_status,
        delta_color="inverse" if escalation_rate > 0.40 else "off",
        help="Percentage of tickets escalated to higher authority"
    )

if fdf.empty:
    st.warning("No records match the current filters. Please adjust your selection.")
    st.stop()

st.markdown("---")

# ---------- Executive Summary ----------
col_left, col_right = st.columns([1, 1])


with col_left:
    st.subheader("Key Findings")
    
    # Multi-tier risk classification
    already_breached_tickets = fdf[fdf["already_breached"] == True]
    
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

    vulnerable_total = len(fdf[fdf["vulnerable_population"] == 1])
    
    num_depts = len(fdf["dept"].unique())
    
    # Calculate escalation stats
    escalated_tickets = fdf[fdf["escalated"] == 1] if "escalated" in fdf.columns else pd.DataFrame()
    escalated_critical = critical_tickets[critical_tickets["escalated"] == 1] if "escalated" in critical_tickets.columns else pd.DataFrame()
    urgent_72h = fdf[(fdf["pred_sla_breach_prob"] > 0.60) & (fdf["is_urgent_72h"] == True)] if "is_urgent_72h" in fdf.columns else pd.DataFrame()
    
    st.markdown(f"""
    **System-wide ({len(fdf)} tickets, {num_depts} departments):**
    - **{len(already_breached_tickets):,} tickets ALREADY BREACHED** (dept-specific SLA passed)
    - **{len(critical_tickets):,} critical tickets** (<24hrs to breach + >80% risk)
    - **{len(escalated_tickets):,} escalated tickets** ({len(escalated_critical):,} critical escalations)
    - **{len(vulnerable_critical)} vulnerable households** in critical tier ({vulnerable_total} total)
    - **{len(urgent_72h):,} urgent for 72h action window** (>60% risk, not yet breached)
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Distribution Analysis",
    "Trend Analysis",
    "Impact Assessment",
    "Action Queue",
    "Monitoring & Oversight",
    "Gemini Assistant"
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

        if len(risk_by_ward) > 0:
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
            st.plotly_chart(fig_ward, width='content', key="ward_risk_breakdown")
        else:
            st.warning("No data available for selected wards")

        st.markdown("---")
        st.markdown("#### District and Department Context")

    # Always show district and department breakdown
    c1, c2 = st.columns(2)

    with c1:
        risk_by_dist = (
            fdf.groupby("district", as_index=False)["pred_sla_breach_prob"]
            .mean().sort_values("pred_sla_breach_prob", ascending=False)
        )
        
        if len(risk_by_dist) > 0:
            max_district = risk_by_dist.iloc[0]["district"]
            max_risk = risk_by_dist.iloc[0]["pred_sla_breach_prob"]

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
            st.plotly_chart(fig1, width='content', key="district_risk_chart")
        else:
            st.info("No district data available with current filters")

    with c2:
        risk_by_dept = (
            fdf.groupby("dept", as_index=False)["pred_sla_breach_prob"]
            .mean().sort_values("pred_sla_breach_prob", ascending=False)
        )
        
        if len(risk_by_dept) > 0:
            max_dept = risk_by_dept.iloc[0]["dept"]

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
            st.plotly_chart(fig2, width='content', key="dept_risk_chart")
        else:
            st.info("No department data available with current filters")

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
    st.plotly_chart(fig3, width='content',  key="category_scatter_chart")

# ====== Tab 2: Trend Analysis ======
with tab2:
    st.subheader("Temporal Patterns and Trends")

    fdf_tab2 = fdf.copy()
    fdf_tab2["day"] = fdf_tab2["created_at"].dt.date
    daily = fdf_tab2.groupby("day", as_index=False).agg(
        tickets=("ticket_id", "count"),
        risk=("pred_sla_breach_prob", "mean"),
        breaches=("sla_breach", "mean")
    )

    t1, t2 = st.columns(2)

    with t1:
        if len(daily) > 0:
            peak_day = daily.loc[daily["tickets"].idxmax(), "day"]
            peak_count = daily["tickets"].max()
        else:
            peak_day = "N/A"
            peak_count = 0

        if len(daily) > 0:
            fig5 = px.line(
                daily,
                x="day",
                y="tickets",
                title=f"Daily Ticket Volume (Peak: {peak_day}, {peak_count} tickets)",
                labels={"day": "Date", "tickets": "Tickets"}
            )
            fig5.update_traces(line_color="#1f77b4")
            st.plotly_chart(fig5, width='content', key="daily_volume_chart")
        else:
            st.info("No trend data available with current filters")

    with t2:
        if len(daily) > 0:
            fig6 = px.line(
                daily,
                x="day",
                y=["risk", "breaches"],
                title="Model Accuracy: Predicted Risk vs Actual Breach Rate",
                labels={"day": "Date", "value": "Rate", "variable": "Metric"},
            )
            fig6.update_traces(name="AI Predicted Risk", selector=dict(name="risk"))
            fig6.update_traces(name="Actual Breaches", selector=dict(name="breaches"))
            st.plotly_chart(fig6, width='content', key="model_accuracy_chart")
        else:
            st.info("No trend data available with current filters")

    st.markdown("---")
    st.markdown("### Escalation Analysis")
    
    if "escalation_level" in fdf.columns:
        esc_col1, esc_col2 = st.columns(2)
        
        with esc_col1:
            st.markdown("#### Escalation Funnel")
            
            # Count tickets at each escalation level
            esc_counts = fdf["escalation_level"].value_counts().sort_index()
            esc_labels = ["Not Escalated", "Supervisor", "District Officer", "Ministry"]
            esc_data = pd.DataFrame({
                "Level": [esc_labels[min(i, 3)] for i in esc_counts.index],
                "Count": esc_counts.values
            })
            
            # Create funnel chart
            fig_esc = px.funnel(
                esc_data,
                y="Level",
                x="Count",
                title="Ticket Escalation Funnel"
            )
            fig_esc.update_layout(height=400)
            st.plotly_chart(fig_esc, width='content', key="escalation_funnel_chart")
            
            # Show escalation rate
            esc_rate = (len(fdf[fdf["escalation_level"] > 0]) / len(fdf) * 100) if len(fdf) > 0 else 0
            st.caption(f"📊 {esc_rate:.1f}% of tickets escalated beyond initial level")
        
        with esc_col2:
            st.markdown("#### Impact on Resolution Time")
            
            # Calculate resolution time by escalation level
            esc_impact = fdf.groupby("escalation_level").agg({
                "resolution_hours": "mean",
                "sla_breach": "mean",
                "ticket_id": "count"
            }).reset_index()
            
            # Map numeric levels to labels
            esc_impact["Level"] = esc_impact["escalation_level"].map({
                0: "None", 
                1: "Supervisor", 
                2: "District", 
                3: "Ministry"
            })
            
            # Create bar chart showing resolution time
            fig_esc_impact = px.bar(
                esc_impact,
                x="Level",
                y="resolution_hours",
                title="Average Resolution Time by Escalation Level",
                labels={"resolution_hours": "Avg Resolution (hours)", "Level": "Escalation Level"},
                color="sla_breach",
                color_continuous_scale=["green", "red"],
                text="ticket_id"
            )
            fig_esc_impact.update_traces(
                texttemplate='%{text} tickets', 
                textposition='outside'
            )
            fig_esc_impact.update_layout(height=400)
            st.plotly_chart(fig_esc_impact, width='content', key="escalation_impact_chart")
            
            # Show insight
            if len(esc_impact) > 1:
                fastest_level = esc_impact.loc[esc_impact["resolution_hours"].idxmin(), "Level"]
                st.caption(f"⚡ Fastest resolution: {fastest_level} escalation level")
    else:
        st.info("ℹ️ Escalation data not available in current dataset. Run updated pipeline to see escalation analysis.")
    
    st.markdown("---")
    # ========== END NEW SECTION ==========
    
    st.markdown("### Weekly Department Trends")
    fdf_tab2["week"] = fdf_tab2["created_at"].dt.to_period("W").dt.start_time
    
    # CREATE THE MISSING weekly_dept DATAFRAME
    weekly_dept = (
        fdf_tab2.groupby(["week", "dept"], as_index=False)["ticket_id"]
        .count().rename(columns={"ticket_id": "tickets"}).sort_values("week")
    )
    
    fig7 = px.area(
        weekly_dept,  # Now it's defined!
        x="week",
        y="tickets",
        color="dept",
        title="Weekly Ticket Volume by Department",
        labels={"week": "Week Starting", "tickets": "Tickets", "dept": "Department"},
    )
    st.plotly_chart(fig7, width='content', key="weekly_dept_trends")
    
    # ========== NEW SECTION: TIME-OF-DAY PATTERNS ==========
    st.markdown("---")
    st.markdown("### Time-of-Day Patterns")
    
    if "created_hour" in fdf_tab2.columns and "is_business_hours" in fdf_tab2.columns:
        tod_col1, tod_col2 = st.columns(2)
        
        with tod_col1:
            st.markdown("#### Hourly Ticket Distribution")
            
            # Group tickets by hour of day
            hourly = fdf_tab2.groupby("created_hour").agg({
                "ticket_id": "count",
                "pred_sla_breach_prob": "mean"
            }).reset_index()
            hourly.columns = ["Hour", "Tickets", "Avg Risk"]
            
            # Create bar chart
            fig_hour = px.bar(
                hourly,
                x="Hour",
                y="Tickets",
                title="Ticket Volume by Hour of Day",
                labels={"Hour": "Hour (0-23)", "Tickets": "Ticket Count"},
                color="Avg Risk",
                color_continuous_scale=["green", "yellow", "red"]
            )
            fig_hour.update_layout(
                xaxis=dict(tickmode='linear', tick0=0, dtick=2),  # Show every 2 hours
                height=400
            )
            st.plotly_chart(fig_hour, width='content', key="hourly_volume_chart")
            
            # Show peak hour insight
            peak_hour = hourly.loc[hourly["Tickets"].idxmax(), "Hour"]
            peak_count = hourly["Tickets"].max()
            st.caption(f"📈 Peak hour: {int(peak_hour):02d}:00 with {peak_count} tickets")
        
        with tod_col2:
            st.markdown("#### Business Hours Impact")
            
            # Compare business hours vs after hours
            bh_stats = fdf_tab2.groupby("is_business_hours").agg({
                "ticket_id": "count",
                "sla_breach": "mean",
                "resolution_hours": "mean"
            }).reset_index()
            
            # Map 0/1 to readable labels
            bh_stats["Period"] = bh_stats["is_business_hours"].map({
                1: "Business Hours\n(9 AM - 5 PM)", 
                0: "After Hours\n(5 PM - 9 AM)"
            })
            
            # Create bar chart for breach rate
            fig_bh = px.bar(
                bh_stats,
                x="Period",
                y="sla_breach",
                title="Breach Rate: Business Hours vs After Hours",
                labels={"sla_breach": "Breach Rate", "Period": ""},
                color="sla_breach",
                color_continuous_scale=["green", "red"],
                text=bh_stats["ticket_id"]
            )
            fig_bh.update_traces(
                texttemplate='%{text} tickets<br>%{y:.1%} breach', 
                textposition='outside'
            )
            fig_bh.update_layout(
                showlegend=False,
                height=400,
                yaxis=dict(tickformat='.0%')
            )
            st.plotly_chart(fig_bh, width='content', key="business_hours_chart")
            
            # Show insight
            if len(bh_stats) == 2:
                bh_breach = bh_stats[bh_stats["is_business_hours"] == 1]["sla_breach"].values[0]
                ah_breach = bh_stats[bh_stats["is_business_hours"] == 0]["sla_breach"].values[0]
                diff = abs(bh_breach - ah_breach)
                
                if ah_breach > bh_breach:
                    st.caption(f"⚠️ After-hours tickets have {diff:.1%} higher breach rate")
                else:
                    st.caption(f"✅ Business hours tickets have {diff:.1%} higher breach rate")
        
        # Additional weekend analysis
        st.markdown("---")
        st.markdown("#### Weekend vs Weekday Comparison")
        
        if "is_weekend" in fdf_tab2.columns:
            weekend_col1, weekend_col2, weekend_col3 = st.columns(3)
            
            weekend_stats = fdf_tab2.groupby("is_weekend").agg({
                "ticket_id": "count",
                "sla_breach": "mean",
                "resolution_hours": "mean"
            }).reset_index()
            
            weekend_stats["Day Type"] = weekend_stats["is_weekend"].map({
                1: "Weekend", 
                0: "Weekday"
            })
            
            with weekend_col1:
                weekend_tickets = weekend_stats[weekend_stats["is_weekend"] == 1]["ticket_id"].values[0] if len(weekend_stats[weekend_stats["is_weekend"] == 1]) > 0 else 0
                weekday_tickets = weekend_stats[weekend_stats["is_weekend"] == 0]["ticket_id"].values[0] if len(weekend_stats[weekend_stats["is_weekend"] == 0]) > 0 else 0
                weekend_pct = (weekend_tickets / (weekend_tickets + weekday_tickets) * 100) if (weekend_tickets + weekday_tickets) > 0 else 0
                
                st.metric(
                    "Weekend Tickets",
                    f"{weekend_tickets:,}",
                    delta=f"{weekend_pct:.1f}% of total"
                )
            
            with weekend_col2:
                weekend_breach = weekend_stats[weekend_stats["is_weekend"] == 1]["sla_breach"].values[0] if len(weekend_stats[weekend_stats["is_weekend"] == 1]) > 0 else 0
                weekday_breach = weekend_stats[weekend_stats["is_weekend"] == 0]["sla_breach"].values[0] if len(weekend_stats[weekend_stats["is_weekend"] == 0]) > 0 else 0
                breach_diff = weekend_breach - weekday_breach
                
                st.metric(
                    "Weekend Breach Rate",
                    f"{weekend_breach:.1%}",
                    delta=f"{breach_diff:+.1%} vs weekday",
                    delta_color="inverse"
                )
            
            with weekend_col3:
                weekend_res = weekend_stats[weekend_stats["is_weekend"] == 1]["resolution_hours"].values[0] if len(weekend_stats[weekend_stats["is_weekend"] == 1]) > 0 else 0
                weekday_res = weekend_stats[weekend_stats["is_weekend"] == 0]["resolution_hours"].values[0] if len(weekend_stats[weekend_stats["is_weekend"] == 0]) > 0 else 0
                res_diff = weekend_res - weekday_res
                
                st.metric(
                    "Weekend Avg Resolution",
                    f"{weekend_res:.1f} hrs",
                    delta=f"{res_diff:+.1f} hrs vs weekday",
                    delta_color="inverse" if res_diff > 0 else "normal"
                )
    else:
        st.info("ℹ️ Time-of-day data not available in current dataset. Run updated pipeline to see temporal analysis.")
    # ========== END NEW SECTION ==========

# ====== Tab 3: Impact Assessment ======
with tab3:
    st.subheader("Resource Planning and Impact Calculator")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Scenario Configuration")
        
        # Staff adjustment
        extra_staff = st.slider(
            "Additional field staff (%)", 
            0, 50, 0, 5, 
            help="Simulate impact of hiring additional staff (diminishing returns apply)"
        )
        
        # Department selection - respect sidebar filters
        available_depts = sorted(fdf["dept"].unique())
        
        if len(available_depts) == 1:
            # Only one department in filtered data - auto-select it
            focus_dept = available_depts[0]
            st.success(f"Analyzing {focus_dept} department ({len(fdf)} tickets in current filter)")
        else:
            # Multiple departments available
            focus_dept = st.selectbox(
                "Target department", 
                available_depts,
                help="Department for resource allocation analysis (from filtered data)"
            )

        st.markdown("---")
        
        # Current state metrics (department-specific AND respecting sidebar filters)
        st.markdown("#### Current State")
        dept_fdf = fdf[fdf["dept"] == focus_dept]  # This now respects sidebar filters
        current_tickets = len(dept_fdf)
        st.info(f"Analyzing {focus_dept} department: {current_tickets} tickets ({current_tickets/len(fdf)*100:.1f}% of total filtered data)")
        current_breach = dept_fdf["sla_breach"].mean() if len(dept_fdf) > 0 else 0
        high_risk_count = len(dept_fdf[dept_fdf["pred_sla_breach_prob"] > 0.60])
        vulnerable_count = len(dept_fdf[dept_fdf["vulnerable_population"] == 1])

        col_a, col_b = st.columns(2)
        col_a.metric("Active Tickets", f"{current_tickets:,}")
        col_b.metric("Current Breach Rate", f"{current_breach:.1%}")
        
        col_c, col_d = st.columns(2)
        col_c.metric("High-Risk Tickets", f"{high_risk_count:,}")
        col_d.metric("Vulnerable Cases", f"{vulnerable_count:,}")


    with col2:
        st.markdown("#### Projected Impact")
        
        if extra_staff > 0 and current_tickets > 0:
            # Calculate ROI using evidence-based model
            roi_data = calculate_roi(
                extra_staff, 
                current_tickets, 
                current_breach, 
                focus_dept
            )
            
            if roi_data is None:
                st.warning(f"Dataset too small for meaningful staff impact analysis. Minimum 20 tickets required (current: {current_tickets}).")
            elif roi_data:
                # Financial metrics
                st.markdown("**Financial Impact**")
                col_e, col_f = st.columns(2)
                
                col_e.metric(
                    "Monthly Cost",
                    f"Rs {roi_data['monthly_cost']:,}",
                    help="Staff salaries + training + 20% overhead"
                )
                
                net_delta = f"Net: Rs {roi_data['net_benefit']:,}"
                col_f.metric(
                    "Monthly Savings",
                    f"Rs {roi_data['monthly_savings']:,}",
                    delta=net_delta,
                    delta_color="normal" if roi_data['net_benefit'] > 0 else "inverse",
                    help="Penalty reduction (Delhi e-SLA model: Rs 10/day)"
                )
                
                col_g, col_h = st.columns(2)
                
                col_g.metric(
                    "ROI",
                    f"{roi_data['roi_percentage']:.1f}%",
                    help="Return on investment percentage"
                )
                
                payback_text = f"{roi_data['payback_months']:.1f} months" if roi_data['payback_months'] != float('inf') else "N/A"
                col_h.metric(
                    "Payback Period",
                    payback_text,
                    help="Time to recover training costs"
                )
                
                # Operational metrics
                st.markdown("**Operational Impact**")
                
                col_i, col_j = st.columns(2)
                
                reduction_delta = f"-{roi_data['reduction_pct']:.1f}% improvement"
                col_i.metric(
                    "Projected Breach Rate",
                    f"{roi_data['new_breach_rate']:.1%}",
                    delta=reduction_delta,
                    delta_color="inverse"
                )
                
                col_j.metric(
                    "Breaches Prevented",
                    f"{roi_data['breaches_prevented']:,}",
                    help="Monthly SLA breach reduction"
                )
                
                # Decision recommendation (compact)
                if roi_data['roi_percentage'] > 50:
                    st.success(f"RECOMMENDED: {roi_data['roi_percentage']:.0f}% ROI (Rs {roi_data['net_benefit']:,}/mo)")
                elif roi_data['roi_percentage'] > 0:
                    st.warning(f"MARGINAL: {roi_data['roi_percentage']:.0f}% ROI")
                else:
                    st.error(f"NOT VIABLE: -{abs(roi_data['roi_percentage']):.0f}% ROI")

        
        else:
            st.info("Adjust the staff percentage slider to see impact projections")
            st.markdown("**Evidence-Based Model:**")
            st.markdown("""
            - Delhi e-SLA penalty structure (Rs 10/day, max Rs 200)
            - Maharashtra govt salary scales (Rs 25K/month)
            - Diminishing returns (20%+ staff increase)
            - Training overhead (20% efficiency loss)
            """)


    st.markdown("---")
    st.markdown("#### Department Performance Comparison")

    if len(fdf) > 0 and len(fdf.groupby("dept")) > 0:
        dept_stats = fdf.groupby("dept").agg({
            "ticket_id": "count",
            "sla_breach": "mean",
            "pred_sla_breach_prob": "mean",
            "resolution_hours": "mean"
        }).round(3)
        dept_stats.columns = ["Total Tickets", "Actual Breach Rate", "AI Risk Score", "Avg Resolution (hrs)"]
        dept_stats = dept_stats.sort_values("AI Risk Score", ascending=False)

        styled_df = dept_stats.style\
            .applymap(color_risk, subset=["AI Risk Score", "Actual Breach Rate"])\
            .format({
                "Actual Breach Rate": "{:.1%}",
                "AI Risk Score": "{:.1%}",
                "Avg Resolution (hrs)": "{:.1f}"
            })
        st.dataframe(styled_df, width='content')
    else:
        st.warning("No department data available for comparison with current filters")

# ====== Tab 4: Action Queue ======
with tab4:
    st.subheader("Priority Queue and Ticket Assignment")

    # 1. Priority weights
    with st.expander("Adjust Priority Weights", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        w = {
            "risk": c1.slider("Risk Weight", 0.0, 1.0, 0.50, 0.01, key="w_risk"),
            "severity": c2.slider("Severity Weight", 0.0, 1.0, 0.20, 0.01, key="w_sev"),
            "vuln": c3.slider("Vulnerability Weight", 0.0, 1.0, 0.15, 0.01, key="w_vuln"),
            "sent": c4.slider("Sentiment Weight", 0.0, 1.0, 0.10, 0.01, key="w_sent"),
            "fest": c5.slider("Festival Weight", 0.0, 1.0, 0.05, 0.01, key="w_fest"),
        }

    # 2. Prepare Data
    fdf_copy = fdf.copy()
    fdf_copy["priority_dynamic"] = compute_dynamic_priority(fdf_copy, w)
    fdf_copy["ministry_default"] = fdf_copy["dept"].map(MINISTRY_MAP).fillna("Cross-Ministry")

    # 3. Create Editable DataFrame (The "Old Way" - Simple)
    # We sort, but we DO NOT try to manually inject previous selections from session state.
    # We let the data_editor handle the persistence during the run.
    editable = fdf_copy.sort_values(
        ["priority_dynamic", "pred_sla_breach_prob"], 
        ascending=[False, False]
    )
    
    # Initialize columns for the editor
    editable.insert(0, "select", False)
    editable.insert(1, "escalate", False)
    editable.insert(2, "route_dept", editable["dept"])
    editable.insert(3, "route_ministry", editable["ministry_default"])

    # 4. Column Configuration
    dept_options = sorted(editable["dept"].unique().tolist()) + ["Cross-Department"]
    
    col_config = {
        "select": st.column_config.CheckboxColumn("Select", default=False),
        "escalate": st.column_config.CheckboxColumn("Escalate", default=False),
        "route_dept": st.column_config.SelectboxColumn("Dept", options=dept_options),
        "route_ministry": st.column_config.SelectboxColumn("Ministry", options=MINISTRY_OPTIONS),
        "pred_sla_breach_prob": st.column_config.NumberColumn("Risk", format="%.0%%"),
        "priority_dynamic": st.column_config.NumberColumn("Priority", format="%.1f"),
    }
    
    display_cols = [
        "select", "escalate", "route_dept", "route_ministry",
        "ticket_id", "created_at", "district", "ward", "dept", 
        "category", "severity", "vulnerable_population", 
        "pred_sla_breach_prob", "priority_dynamic", "historical_backlog"
    ]

    # 5. Render Data Editor
    st.markdown("### Ticket Queue")
    
    # Using a unique key helps Streamlit maintain state
    edited_df = st.data_editor(
        editable[display_cols].head(100),
        hide_index=True,
        use_container_width=True,
        column_config=col_config,
        disabled=["ticket_id", "created_at", "district", "pred_sla_breach_prob", "priority_dynamic"],
        key="final_queue_editor_v2" 
    )

    # 6. Extract Selected Rows directly from the Editor Output
    # This is the "Old Version" logic that works reliably
    sel = edited_df[edited_df["select"]]
    
    # Show selection count
    if len(sel) > 0:
        st.success(f"✅ {len(sel)} tickets selected")

    # 7. Action Buttons
    col_a, col_b, col_c = st.columns([1, 1, 2])
    
    if col_b.button("Assign Selected", disabled=len(sel) == 0, type="primary"):
        for _, r in sel.iterrows():
            payload = {
                "ticket_id": r["ticket_id"],
                "assignment": {
                    "ministry": r["route_ministry"], 
                    "department": r["route_dept"],
                    "escalated": bool(r["escalate"])
                },
                "status": "In Progress",
                "assigned_at": datetime.now().isoformat(),
                "risk_score": float(r["pred_sla_breach_prob"]),
                "priority_score": float(r["priority_dynamic"]),
                # Metadata for dashboard
                "ministry": r["route_ministry"],
                "department": r["route_dept"]
            }
            # Save to Cloud
            save_assignment_firestore(payload)
        
        st.success(f"✅ Successfully assigned {len(sel)} tickets! Data saved to Firestore.")
        time.sleep(1) # Give DB time to write
        st.rerun()
        
        # Optional: Clear history button logic logic can go here or be handled by manual uncheck
        # Note: In this simple mode, boxes stay checked until user unchecks or reloads page
        # This is often preferred behavior for batch operations.

    if col_c.button("Clear Selection"):
        # This forces a cache clear to reset the editor
        st.rerun()

    # Assignment Log
    if "assignments" in st.session_state and st.session_state["assignments"]:
        st.markdown("---")
        st.markdown("### Assignment History")
        st.dataframe(pd.DataFrame(st.session_state["assignments"]).tail(5))



# ====== Tab 5: Monitoring & Oversight ======
with tab5:
    st.subheader("Assignment Monitoring and Ministry Performance")

    # Get assigned tickets from session state
    assignments_data = get_assignments_firestore()

    if assignments_data and len(assignments_data) > 0:
        assigned_df = pd.DataFrame(assignments_data)

        # Handle potential missing columns if DB is fresh
        if "ministry" not in assigned_df.columns:
             assigned_df["ministry"] = assigned_df["assignment"].apply(lambda x: x.get("ministry", "Unknown") if isinstance(x, dict) else "Unknown")
        if "department" not in assigned_df.columns:
             assigned_df["department"] = assigned_df["assignment"].apply(lambda x: x.get("department", "Unknown") if isinstance(x, dict) else "Unknown")
        
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
            width='content',
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
            st.plotly_chart(fig_ministry, width='content', key="ministry_distribution_chart")

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
            st.plotly_chart(fig_time, width='content', key="ministry_response_time_chart")

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
            width='content',
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

# ====== Tab 6: Gemini Assistant ======
with tab6:
    st.subheader("🤖 GenAI Citizen & Officer Assistant (Powered by Vertex AI)")
    
    col_chat, col_context = st.columns([3, 1])
    
    with col_context:
        st.info("ℹ️ **Live Context:** Connected to BigQuery & Vertex AI")
        st.markdown(f"**Current Dataset:**")
        st.markdown(f"- **Tickets:** {len(fdf)}")
        st.markdown(f"- **Avg Risk:** {fdf['pred_sla_breach_prob'].mean():.1%}")
        
        # Dynamic top category based on filters
        if not fdf.empty:
            top_cat = fdf['category'].mode()[0]
            st.markdown(f"- **Top Issue:** {top_cat}")
            
        st.markdown("---")
        st.markdown("**Sample Queries:**")
        st.code("Summarize critical water issues", language=None)
        st.code("Which district has the highest risk?", language=None)
        st.code("Draft an escalation email for the top risk ticket", language=None)

    with col_chat:
        # Create a fixed-height scrollable container for the history
        chat_container = st.container(height=500, border=True)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Render chat history INSIDE the container
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Input field stays OUTSIDE the container (at the bottom)
        if prompt := st.chat_input("Ask Gemini about the filtered data..."):
            # 1. Add user message to state
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # 2. Render user message immediately in container
            with chat_container:
                st.chat_message("user").markdown(prompt)
                
                # 3. Generate and render response
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing dashboard data..."):
                        response = get_gemini_response(prompt, fdf)
                        st.markdown(response)
            
            # 4. Add assistant response to state
            st.session_state.messages.append({"role": "assistant", "content": response})

st.caption("AI-Powered Governance Platform • Advanced Analytics for Public Service Management")