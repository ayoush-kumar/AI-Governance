import os, uuid, random, math, json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
import joblib


def make_dirs():
    base_dir = "governance_prototype"
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def synthesize(base_dir, n=5000, days=180, seed=42):
    np.random.seed(seed); random.seed(seed)
    start_date = datetime.today() - timedelta(days=days)

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
    severity_map = {"clinic_wait":2,"vector_control":3,"drug_stockout":3,"pothole":2,"footpath_damage":1,"signal_fault":3,
                    "pipe_leak":2,"low_pressure":2,"water_quality":3,"school_infra":2,"midday_meal":3,"teacher_absence":2,
                    "streetlight_out":2,"encroachment":2,"fire_safety":3}
    district_pressure = {"Mumbai City":0.8,"Mumbai Suburban":1.0,"Pune":0.9,"Thane":0.95,"Nagpur":0.7,
                         "Nashik":0.6,"Aurangabad":0.65,"Solapur":0.55,"Amravati":0.5,"Kolhapur":0.45}

    rows = []
    for _ in range(n):
        created_at = start_date + timedelta(days=np.random.randint(0, days))
        district = random.choice(districts)
        ward = random.choice(wards)
        dept = random.choice(depts)
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

        logit = (
            -2.0
            + 0.8*severity
            + 1.2*vulnerable_population
            + 0.8*(-citizen_sentiment)
            + 0.01*historical_backlog
            + 1.0*(1 - resource_availability)
            + 0.005*weather_rain_mm
            + 0.006*traffic_index
            + 0.6*district_pressure[district]
            + 0.5*fes_event
            + np.random.normal(0, 0.5)
        )
        p_breach = 1/(1+math.exp(-logit))
        p_breach = min(max(p_breach, 0.001), 0.999)
        sla_breach = np.random.binomial(1, p_breach)

        base_resolution = 24 + 18*severity + 30*vulnerable_population
        resolution_hours = max(1, np.random.normal(loc=base_resolution + 60*p_breach, scale=12))

        rows.append({
            "ticket_id": str(uuid.uuid4()),
            "created_at": created_at.date().isoformat(),
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
            "sla_breach": sla_breach,
            "resolution_hours": float(resolution_hours)
        })

    df = pd.DataFrame(rows)
    data_path = os.path.join(base_dir, "synthetic_tickets.csv")
    df.to_csv(data_path, index=False)
    return df, data_path


def train_and_score(df, base_dir):
    feature_cols = ["district","ward","dept","category","reported_via",
                    "severity","citizen_sentiment","vulnerable_population",
                    "historical_backlog","resource_availability","weather_rain_mm",
                    "traffic_index","festival_season"]
    target = "sla_breach"

    X = df[feature_cols].copy()
    y = df[target].astype(int).values

    cat_cols = ["district","ward","dept","category","reported_via"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols)
        ]
    )
    clf = GradientBoostingClassifier(random_state=42)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)

    model_path = os.path.join(base_dir, "sla_breach_model.joblib")
    joblib.dump(pipe, model_path)

    # Score full dataset and compute priority signal for downstream use
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
    cols = ["ticket_id","created_at","district","ward","dept","category","severity",
            "citizen_sentiment","vulnerable_population","historical_backlog","resource_availability",
            "weather_rain_mm","traffic_index","festival_season",
            "sla_breach","pred_sla_breach_prob","priority_score","reported_via","resolution_hours"]
    df[cols].to_csv(pred_path, index=False)

    # Model explainability for offline analysis
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

    return {
        "auc": float(auc),
        "brier": float(brier),
        "model_path": model_path,
        "predictions": pred_path,
        "fi_path": fi_path
    }


def main():
    base_dir = make_dirs()
    df, data_path = synthesize(base_dir)
    metrics = train_and_score(df, base_dir)

    print(json.dumps({
        "base_dir": base_dir,
        "data": data_path,
        "model": metrics["model_path"],
        "predictions": metrics["predictions"],
        "feature_importance": metrics["fi_path"],
        "auc": metrics["auc"],
        "brier": metrics["brier"]
    }, indent=2))


if __name__ == "__main__":
    main()
