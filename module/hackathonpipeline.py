import os, uuid, random, math, json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
import joblib


# ---------- SLA Configuration (Based on Delhi e-SLA Research) ----------
SLA_HOURS_BY_DEPT = {
    "Health": 7 * 24,      # 7 days (168 hours) - Critical health services
    "Roads": 15 * 24,      # 15 days (360 hours) - Infrastructure repair
    "Water": 10 * 24,      # 10 days (240 hours) - Water supply issues
    "Education": 30 * 24,  # 30 days (720 hours) - Non-emergency
    "Safety": 7 * 24       # 7 days (168 hours) - Public safety priority
}

# Escalation thresholds (based on Maharashtra research)
ESCALATION_LEVELS = {
    0: "None",
    1: "Supervisor",
    2: "District Officer", 
    3: "Ministry"
}


def make_dirs():
    base_dir = "governance_prototype"
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def synthesize(base_dir, n=5000, days=180, seed=42):
    """Generate synthetic ticket data with realistic patterns"""
    np.random.seed(seed)
    random.seed(seed)
    
    # Generate data from last 180 days (May 30 to Nov 26, 2025)
    end_date = datetime(2025, 11, 26)  # Current date
    start_date = end_date - timedelta(days=days)

    districts = ["Mumbai City","Mumbai Suburban","Pune","Thane","Nagpur","Nashik","Aurangabad","Solapur","Amravati","Kolhapur"]
    wards = [f"W{str(i).zfill(2)}" for i in range(1, 31)]
    depts = ["Health","Roads","Water","Education","Safety"]
    categories = {
        "Health": ["clinic_wait","vector_control","drug_stockout"],
        "Roads": ["pothole","footpath_damage","signal_fault"],
        "Water": ["pipe_leak","low_pressure","water_quality"],
        "Education": ["school_infra","midday_meal","teacher_absence"],
        "Safety": ["streetlight_out","encroachment","fire_safety"]
    }
    channels = ["app","call","walk-in"]
    festival_months = {8,9,10,11}
    severity_map = {
        "clinic_wait":2,"vector_control":3,"drug_stockout":3,
        "pothole":2,"footpath_damage":1,"signal_fault":3,
        "pipe_leak":2,"low_pressure":2,"water_quality":3,
        "school_infra":2,"midday_meal":3,"teacher_absence":2,
        "streetlight_out":2,"encroachment":2,"fire_safety":3
    }
    district_pressure = {
        "Mumbai City":0.8,"Mumbai Suburban":1.0,"Pune":0.9,"Thane":0.95,"Nagpur":0.7,
        "Nashik":0.6,"Aurangabad":0.65,"Solapur":0.55,"Amravati":0.5,"Kolhapur":0.45
    }
    
    # Department-specific ticket volume distribution (realistic patterns)
    dept_weights = {
        "Health": 0.22,      # 25% of tickets (1,250)
        "Roads": 0.26,       # 30% of tickets (1,500) - most common
        "Water": 0.20,       # 20% of tickets (1,000)
        "Education": 0.17,   # 15% of tickets (750)
        "Safety": 0.14       # 10% of tickets (500)
    }

    rows = []
    
    # Generate tickets with department stratification
    tickets_per_dept = {
        dept: int(n * weight) for dept, weight in dept_weights.items()
    }
    
    # Adjust for rounding to ensure exactly n tickets
    total_assigned = sum(tickets_per_dept.values())
    tickets_per_dept["Roads"] += (n - total_assigned)
    
    print(f"\n Generating {n} tickets with department distribution:")
    for dept, count in tickets_per_dept.items():
        print(f"  - {dept}: {count} tickets ({count/n*100:.1f}%)")
    
    # Generate tickets per department
    for dept in depts:
        dept_ticket_count = tickets_per_dept[dept]
        
        for _ in range(dept_ticket_count):
            # Generate timestamp with hour component
            # Research-based: Delhi e-SLA shows 30% tickets < 7 days, 65% < 30 days
            rand = np.random.rand()
            if rand < 0.30:
                # Tier 1: Last 7 days (30%) - Daily fresh tickets
                days_offset = days - np.random.randint(0, 7)

            elif rand < 0.65:
                # Tier 2: 8-30 days (35%) - Active working set
                days_offset = days - np.random.randint(8, 30)

            elif rand < 0.90:
                # Tier 3: 31-90 days (25%) - Recent backlog
                days_offset = days - np.random.randint(31, 90)
                
            else:
                # Tier 4: 91-180 days (10%) - Old backlog/complex cases
                days_offset = days - np.random.randint(91, 180)

            hour = np.random.randint(0, 24)
            minute = np.random.randint(0, 60)
            created_at = start_date + timedelta(days=days_offset, hours=hour, minutes=minute)
            
            # Time-based features
            created_hour = created_at.hour
            is_business_hours = 1 if 9 <= created_hour <= 17 else 0
            is_weekend = 1 if created_at.weekday() >= 5 else 0
            
            district = random.choice(districts)
            ward = random.choice(wards)
            category = random.choice(categories[dept])
            channel = random.choice(channels)
            severity = severity_map[category]
            citizen_sentiment = np.clip(np.random.normal(loc=0.0, scale=0.6), -1, 1)
            vulnerable_population = np.random.binomial(1, 0.25 if dept in ["Health","Education"] else 0.15)
            historical_backlog = max(0, int(np.random.normal(loc=60, scale=30)))
            resource_availability = np.clip(np.random.beta(5,2), 0, 1)
            weather_rain_mm = max(0, np.random.gamma(shape=2, scale=15)) if created_at.month in {6,7,8,9} else np.random.gamma(1.2, 5)
            traffic_index = int(np.clip(np.random.normal(loc=55, scale=20), 0, 100))
            fes_event = 1 if created_at.month in festival_months and np.random.rand() < 0.25 else 0

            # Calculate breach probability using logit model
            logit = (
                -1.5  # Baseline adjusted for ~30-35% breach rate
                + 0.8 * severity
                + 1.2 * vulnerable_population
                + 0.8 * (-citizen_sentiment)
                + 0.01 * historical_backlog
                + 1.0 * (1 - resource_availability)
                + 0.005 * weather_rain_mm
                + 0.006 * traffic_index
                + 0.6 * district_pressure[district]
                + 0.5 * fes_event
                + 0.4 * (1 - is_business_hours)  # After-hours penalty
                + 0.5 * is_weekend  # Weekend delay
                + np.random.normal(0, 0.5)
            )

            p_breach = 1 / (1 + math.exp(-logit))
            p_breach = min(max(p_breach, 0.001), 0.999)
            
            # Department-specific SLA deadline
            sla_deadline_hours = SLA_HOURS_BY_DEPT[dept]

            # Simplified resolution model that ensures realistic breach distribution
            # Base resolution factors
            base_hours = 48  # 2 days baseline
            severity_factor = severity * 24  # +24/48/72 hours based on severity (1/2/3)
            vuln_factor = vulnerable_population * 36  # +36 hours if vulnerable
            backlog_factor = (historical_backlog / 100) * 24  # +24 hours at max backlog
            resource_factor = (1 - resource_availability) * 48  # Up to +48 hours if low resources
            
            # Time-based delays
            after_hours_delay = (1 - is_business_hours) * 12  # +12 hours after hours
            weekend_delay = is_weekend * 24  # +24 hours on weekend
            
            # Weather and traffic
            weather_delay = (weather_rain_mm / 50) * 12  # Up to +12 hours in heavy rain
            traffic_delay = (traffic_index / 100) * 6  # Up to +6 hours in traffic
            
            # Calculate total expected resolution time
            expected_resolution = (
                base_hours 
                + severity_factor 
                + vuln_factor 
                + backlog_factor 
                + resource_factor
                + after_hours_delay 
                + weekend_delay
                + weather_delay
                + traffic_delay
            )
            
            # Add random variation (normal distribution)
            resolution_hours = max(24, np.random.normal(loc=expected_resolution, scale=expected_resolution * 0.3))
            
            # Use p_breach to add realistic delay pattern
            # High p_breach → ticket likely takes longer
            breach_multiplier = 1.0 + (p_breach * 0.8)  # 1.0 to 1.8x multiplier
            resolution_hours *= breach_multiplier
            
            # Determine SLA breach
            sla_breach = 1 if resolution_hours > sla_deadline_hours else 0
            
            # Cap at 3x SLA to keep realistic
            resolution_hours = min(resolution_hours, sla_deadline_hours * 3)

            # Escalation logic (NO DATA LEAKAGE - uses p_breach, not sla_breach)
            p_escalate = np.clip(
                0.1 +  # Base escalation rate
                0.3 * (p_breach > 0.6) +  # High-risk tickets escalate more
                0.2 * (-citizen_sentiment) +  # Angry citizens escalate more
                0.15 * (severity / 3.0),  # Severe issues escalate more
                0, 0.7  # Max 70% escalation probability
            )
            
            escalated = np.random.binomial(1, p_escalate)
            escalation_level = 0  # 0=none, 1=supervisor, 2=district, 3=ministry
            
            if escalated:
                # Distribution: 60% supervisor, 30% district, 10% ministry
                escalation_level = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
                
                # Escalated tickets get faster resolution (reduction factor)
                escalation_speedup = 0.7 ** escalation_level  # Level 1: 70%, Level 2: 49%, Level 3: 34%
                resolution_hours *= escalation_speedup
                
                # Recalculate breach status after escalation
                sla_breach = 1 if resolution_hours > sla_deadline_hours else 0
            
            escalation_days_to_resolve = resolution_hours / 24

            rows.append({
                "ticket_id": str(uuid.uuid4()),
                "created_at": created_at.isoformat(),
                "district": district,
                "ward": ward,
                "dept": dept,
                "category": category,
                "reported_via": channel,
                "severity": severity,
                "citizen_sentiment": citizen_sentiment,
                "vulnerable_population": vulnerable_population,
                "historical_backlog": historical_backlog,
                "resource_availability": resource_availability,
                "weather_rain_mm": float(weather_rain_mm),
                "traffic_index": traffic_index,
                "festival_season": fes_event,
                # Time features
                "created_hour": created_hour,
                "is_business_hours": is_business_hours,
                "is_weekend": is_weekend,
                # SLA features
                "sla_deadline_hours": sla_deadline_hours,
                "sla_breach": sla_breach,
                "resolution_hours": float(resolution_hours),
                # Escalation features (metadata only, not for model training)
                "escalated": escalated,
                "escalation_level": escalation_level,
                "escalation_days": float(escalation_days_to_resolve)
            })

    df = pd.DataFrame(rows)
    
    # Verify department distribution
    print(f"\n✓ Generated ticket distribution:")
    dept_dist = df['dept'].value_counts().sort_index()
    for dept, count in dept_dist.items():
        print(f"  - {dept}: {count} tickets ({count/len(df)*100:.1f}%)")
    
    # Verify time distribution
    df_temp = df.copy()
    df_temp['created_dt'] = pd.to_datetime(df_temp['created_at'])
    df_temp['days_ago'] = (end_date - df_temp['created_dt']).dt.days
    recent_count = len(df_temp[df_temp['days_ago'] <= 30])
    older_count = len(df_temp[df_temp['days_ago'] > 30])
    
    print(f"\n✓ Time distribution:")
    print(f"  - Last 30 days: {recent_count} tickets ({recent_count/len(df)*100:.1f}%)")
    print(f"  - 31-180 days ago: {older_count} tickets ({older_count/len(df)*100:.1f}%)")
    
    data_path = os.path.join(base_dir, "synthetic_tickets.csv")
    df.to_csv(data_path, index=False)
    return df, data_path


def train_and_score(df, base_dir):
    """Train model and evaluate performance"""
    # Features for model (NO escalation - that's data leakage)
    feature_cols = [
        "district","ward","dept","category","reported_via",
        "severity","citizen_sentiment","vulnerable_population",
        "historical_backlog","resource_availability","weather_rain_mm",
        "traffic_index","festival_season",
        "created_hour","is_business_hours","is_weekend"
    ]
    target = "sla_breach"

    X = df[feature_cols].copy()
    y = df[target].astype(int).values

    cat_cols = ["district","ward","dept","category","reported_via"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols)
        ]
    )
    
    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    
    # Model validation
    train_breach_rate = y_train.mean()
    test_breach_rate = y_test.mean()
    high_risk_breach = y_test[probs > 0.7].mean() if sum(probs > 0.7) > 0 else 0
    low_risk_breach = y_test[probs < 0.3].mean() if sum(probs < 0.3) > 0 else 0
    
    print(f"\n✓ Model Validation:")
    print(f"  - Training breach rate: {train_breach_rate:.1%}")
    print(f"  - Test breach rate: {test_breach_rate:.1%}")
    print(f"  - High-risk (>70%) breach rate: {high_risk_breach:.1%}")
    print(f"  - Low-risk (<30%) breach rate: {low_risk_breach:.1%}")
    print(f"  - AUC: {auc:.3f}")
    print(f"  - Brier Score: {brier:.3f}")

    model_path = os.path.join(base_dir, "sla_breach_model.joblib")
    joblib.dump(pipe, model_path)

    # Score full dataset
    df["pred_sla_breach_prob"] = pipe.predict_proba(X)[:, 1]
    priority = (
        0.5*df["pred_sla_breach_prob"]
        + 0.2*(df["severity"]/3.0)
        + 0.15*(df["vulnerable_population"])
        + 0.1*((-df["citizen_sentiment"]+1)/2.0)
        + 0.05*(df["festival_season"])
    )
    df["priority_score"] = (100*np.clip(priority, 0, 1)).round(1)

    pred_path = os.path.join(base_dir, "predictions.csv")
    cols = [
        "ticket_id","created_at","district","ward","dept","category","severity",
        "citizen_sentiment","vulnerable_population","historical_backlog","resource_availability",
        "weather_rain_mm","traffic_index","festival_season",
        "created_hour","is_business_hours","is_weekend",
        "sla_deadline_hours","sla_breach","resolution_hours",
        "escalated","escalation_level","escalation_days",
        "pred_sla_breach_prob","priority_score","reported_via"
    ]
    df[cols].to_csv(pred_path, index=False)

    # Feature importance
    perm = permutation_importance(
        pipe, X_test, y_test, n_repeats=5, random_state=42, scoring="roc_auc"
    )
    fi = []
    for i, col in enumerate(feature_cols):
        fi.append({
            "feature": col,
            "importance_mean": float(perm.importances_mean[i]),
            "importance_std": float(perm.importances_std[i])
        })
    fi_df = pd.DataFrame(fi).sort_values("importance_mean", ascending=False)
    fi_path = os.path.join(base_dir, "feature_importance.csv")
    fi_df.to_csv(fi_path, index=False)

    print(f"\n✓ Top 10 Most Important Features:")
    for idx, row in fi_df.head(10).iterrows():
        print(f"  {idx+1}. {row['feature']}: {row['importance_mean']:.4f} (±{row['importance_std']:.4f})")
    
    # Data generation consistency check
    LOGIT_COEFFICIENTS = {
        "severity": 0.8, "vulnerable_population": 1.2, "citizen_sentiment": 0.8,
        "resource_availability": 1.0, "historical_backlog": 0.01, "weather_rain_mm": 0.005,
        "traffic_index": 0.006, "festival_season": 0.5, "is_business_hours": 0.4,
        "is_weekend": 0.5, "district": 0.6, "dept": 0.0, "category": 0.0,
        "ward": 0.0, "reported_via": 0.0, "created_hour": 0.0
    }
    
    print(f"\n✓ Data Generation Consistency Check:")
    print(f"  {'Feature':<30} {'Logit Coef':<12} {'Model Rank':<12} {'Importance':<12} {'Status'}")
    print(f"  {'-'*85}")
    
    high_coef_features = sorted(
        [(f, c) for f, c in LOGIT_COEFFICIENTS.items() if c >= 0.5],
        key=lambda x: x[1], reverse=True
    )
    
    consistency_issues = 0
    for feat, coef in high_coef_features:
        feat_row = fi_df[fi_df['feature'] == feat]
        if not feat_row.empty:
            rank = feat_row.index[0] + 1
            importance = feat_row['importance_mean'].values[0]
            status = "✓ PASS" if rank <= 7 else "✗ FAIL"
            if rank > 7:
                consistency_issues += 1
            print(f"  {feat:<30} {coef:<12.2f} #{rank:<11} {importance:<12.4f} {status}")

    print(f"\n{'='*85}")
    if consistency_issues == 0:
        print(f"✓ PASS: Model learned patterns consistent with data generation")
    else:
        print(f"⚠️  WARNING: {consistency_issues} consistency issue(s) detected")
    print(f"{'='*85}\n")

    return {
        "auc": float(auc),
        "brier": float(brier),
        "model_path": model_path,
        "predictions": pred_path,
        "fi_path": fi_path
    }


def validate_dashboard_readiness(df):
    """Validate dashboard metrics"""
    print(f"\n{'='*60}")
    print(f"DASHBOARD READINESS CHECK")
    print(f"{'='*60}")
    
    now = datetime(2025, 11, 26)
    df['created_dt'] = pd.to_datetime(df['created_at'])
    df['hours_since_creation'] = (now - df['created_dt']).dt.total_seconds() / 3600
    df['hours_until_breach_dashboard'] = 72 - df['hours_since_creation']
    df['already_breached_dashboard'] = df['hours_until_breach_dashboard'] < 0
    
    already_breached_count = df['already_breached_dashboard'].sum()
    critical_tickets = len(df[
        (df['pred_sla_breach_prob'] > 0.80) & 
        (df['hours_until_breach_dashboard'] < 24) &
        (df['hours_until_breach_dashboard'] > 0)
    ])
    dire_tickets = len(df[
        (df['pred_sla_breach_prob'] > 0.80) &
        (df['hours_until_breach_dashboard'].between(24, 72))
    ])
    high_risk = len(df[(df['pred_sla_breach_prob'] > 0.60) & (df['pred_sla_breach_prob'] <= 0.80)])
    
    total_tickets = len(df)
    already_breached_pct = already_breached_count / total_tickets * 100
    
    print(f"\nExpected Dashboard Metrics:")
    print(f"  Total Tickets: {total_tickets:,}")
    print(f"  Already Breached: {already_breached_count:,} ({already_breached_pct:.1f}%)")
    print(f"  Critical (<24hrs): {critical_tickets:,}")
    print(f"  Dire (24-72hrs): {dire_tickets:,}")
    print(f"  High Risk (60-80%): {high_risk:,}")
    
    if already_breached_pct > 60:
        print(f"\n❌ FAIL: Too many already breached ({already_breached_pct:.1f}% > 60%)")
    elif already_breached_pct < 20:
        print(f"\n⚠️  WARNING: Too few already breached ({already_breached_pct:.1f}% < 20%)")
    else:
        print(f"\n✓ PASS: Already breached rate is realistic ({already_breached_pct:.1f}%)")
    
    if critical_tickets == 0 and dire_tickets == 0:
        print(f"\n⚠️  WARNING: No urgent tickets for dashboard")
    else:
        print(f"\n✓ PASS: Dashboard will have actionable urgent tickets")
    
    print(f"\n{'='*60}\n")
    
    return {
        "already_breached_pct": already_breached_pct,
        "critical_count": critical_tickets,
        "dire_count": dire_tickets,
        "high_risk_count": high_risk
    }


def main():
    base_dir = make_dirs()
    df, data_path = synthesize(base_dir)
    
    # Debug check
    breach_rate = df['sla_breach'].mean()
    print(f"\n{'='*60}")
    print(f"SYNTHETIC DATA CHECK")
    print(f"{'='*60}")
    print(f"Total tickets: {len(df):,}")
    print(f"Breach rate: {breach_rate:.1%}")
    print(f"Breaches: {df['sla_breach'].sum():,}")
    print(f"Non-breaches: {(df['sla_breach']==0).sum():,}")
    
    if breach_rate == 0.0 or breach_rate == 1.0:
        print(f"\n❌ FATAL: {breach_rate:.0%} breach rate - cannot train model")
        return
    else:
        print(f"\n✓ PASS: Breach rate is realistic ({breach_rate:.1%})")
    print(f"{'='*60}\n")
    
    metrics = train_and_score(df, base_dir)
    pred_df = pd.read_csv(metrics["predictions"])
    dashboard_metrics = validate_dashboard_readiness(pred_df)

    print(json.dumps({
        "base_dir": base_dir,
        "data": data_path,
        "model": metrics["model_path"],
        "predictions": metrics["predictions"],
        "feature_importance": metrics["fi_path"],
        "auc": metrics["auc"],
        "brier": metrics["brier"],
        "dashboard_readiness": dashboard_metrics
    }, indent=2))


if __name__ == "__main__":
    main()
