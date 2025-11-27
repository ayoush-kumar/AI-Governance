Here is the formal technical documentation for the migration of the **AI-Powered Governance Platform** from a local prototype to a Cloud-Native architecture on Google Cloud Platform (GCP).

---

# AI Governance Platform: Cloud Migration & Deployment Report

**Project:** AI-Powered Governance Platform (Maharashtra Government Prototype)  
**Target Environment:** Google Cloud Platform (Cloud Run, BigQuery, Firestore, Vertex AI)  
**Deployment Region:** `asia-south1` (Mumbai)  
**AI Inference Region:** `us-central1` (Vertex AI Global Endpoint)

---

## 1. Cloud Infrastructure Provisioning

The following commands were executed to initialize the environment, create data resources, and grant necessary permissions (IAM).

### 1.1 Enable APIs
```bash
gcloud services enable \
  bigquery.googleapis.com \
  firestore.googleapis.com \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  --project hackathon-pipeline
```

### 1.2 Resource Creation
*   **BigQuery (Data Warehouse):**
    ```bash
    bq --location=asia-south1 mk -d \
      --description "Governance Dashboard Data" \
      hackathon-pipeline:governance_data
    ```
*   **Firestore (Application Database):**
    *   Created via Console in `asia-south1` (Native Mode).
    *   Collection initialized: `assignments`.
*   **Artifact Registry (Docker Repository):**
    *   Repo Name: `governance-app`
    *   Location: `asia-south1`

### 1.3 IAM Permission Assignment
Granted the default Compute Service Account access to managed resources.

```bash
# Retrieve Project Number
PROJECT_NUM=$(gcloud projects describe hackathon-pipeline --format="value(projectNumber)")

# Grant BigQuery Data Editor
gcloud projects add-iam-policy-binding hackathon-pipeline \
    --member=serviceAccount:${PROJECT_NUM}-compute@developer.gserviceaccount.com \
    --role=roles/bigquery.dataEditor

# Grant Firestore User
gcloud projects add-iam-policy-binding hackathon-pipeline \
    --member=serviceAccount:${PROJECT_NUM}-compute@developer.gserviceaccount.com \
    --role=roles/datastore.user

# Grant Vertex AI User
gcloud projects add-iam-policy-binding hackathon-pipeline \
    --member=serviceAccount:${PROJECT_NUM}-compute@developer.gserviceaccount.com \
    --role=roles/aiplatform.user
```

---

## 2. Data Migration Strategy (ETL)

We transitioned from local CSV files to Google BigQuery to simulate a scalable data lake.

**Script Used:** `upload_to_cloud.py`
**Execution:** Ran once locally to seed the cloud database.

```python
import pandas as pd
from google.cloud import bigquery

# Config
PROJECT_ID = "hackathon-pipeline"
TABLE_REF = "hackathon-pipeline.governance_data.predictions"
CSV_PATH = "governance_prototype/predictions.csv"

# Logic
client = bigquery.Client(project=PROJECT_ID)
df = pd.read_csv(CSV_PATH)
df["created_at"] = pd.to_datetime(df["created_at"])

job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
job = client.load_table_from_dataframe(df, TABLE_REF, job_config=job_config)
job.result()
print(f"✅ Migrated {len(df)} records to BigQuery.")
```

---

## 3. Codebase Refactoring

The following critical changes were made to `dashboard_streamlit.py` to integrate cloud services.

### 3.1 Imports & Initialization
Added Google Cloud client libraries.

```python
import uuid
import time
from google.cloud import bigquery
from google.cloud import firestore
import vertexai
from vertexai.generative_models import GenerativeModel

PROJECT_ID = "hackathon-pipeline"
REGION = "asia-south1"
db = firestore.Client(project=PROJECT_ID) # Firestore Client
```

### 3.2 Data Loading (BigQuery Integration)
Replaced local CSV reading with a cached BigQuery SQL query.

```python
@st.cache_data(ttl=300)
def load_predictions() -> pd.DataFrame:
    client = bigquery.Client(project=PROJECT_ID)
    query = f"SELECT * FROM `{PROJECT_ID}.governance_data.predictions`"
    df = client.query(query).to_dataframe()
    
    # Type casting for stability
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["pred_sla_breach_prob"] = pd.to_numeric(df["pred_sla_breach_prob"], errors="coerce").fillna(0.0)
    df["priority_score"] = pd.to_numeric(df["priority_score"], errors="coerce").fillna(0.0)
    return df
```

### 3.3 Persistence Logic (Firestore Integration)
Replaced `st.session_state` for ticket assignment with Firestore to ensure data persistence across container restarts.

```python
def save_assignment_firestore(data):
    doc_id = str(uuid.uuid4())
    data['saved_at'] = firestore.SERVER_TIMESTAMP
    db.collection('assignments').document(doc_id).set(data)

def get_assignments_firestore():
    docs = db.collection('assignments').order_by('saved_at', direction=firestore.Query.DESCENDING).limit(50).stream()
    return [doc.to_dict() for doc in docs]
```

### 3.4 GenAI Implementation (Vertex AI)
Implemented `gemini-2.0-flash-lite` with **Context Injection** to solve the "I don't know" issue. Note the use of `us-central1` to access the specific model version.

```python
def get_gemini_response(query, context_df):
    try:
        # Cross-region call to US-Central1 for model availability
        vertexai.init(project=PROJECT_ID, location="us-central1")
        model = GenerativeModel("gemini-2.0-flash-lite")
        
        # Context Injection (Top 25 rows + Aggregates)
        sample_csv = context_df.sort_values("pred_sla_breach_prob", ascending=False).head(25).to_string(index=False)
        stats = f"Total Tickets: {len(context_df)}, Avg Risk: {context_df['pred_sla_breach_prob'].mean():.1%}"
        
        prompt = f"""
        Role: Government Analytics Assistant.
        Data Context: {stats}
        Sample Critical Data: {sample_csv}
        User Query: {query}
        Task: Answer based strictly on the data provided.
        """
        
        return model.generate_content(prompt).text
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"
```

### 3.5 UI Enhancements
*   **Tab 4 (Action Queue):** Updated to write to Firestore.
*   **Tab 6 (Chat Interface):** Implemented `st.container(height=500)` to fix scrolling issues and keep the input bar fixed at the bottom.

---

## 4. Containerization & Deployment

The application was containerized using Docker and deployed to Google Cloud Run as a serverless service.

### 4.1 Dockerfile Configuration
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["streamlit", "run", "Dashboard/dashboard_streamlit.py", "--server.port=8080", "--server.address=0.0.0.0"]
```

### 4.2 Deployment Commands
These are the final commands used to deploy the working version.

**1. Authentication:**
```bash
gcloud auth configure-docker asia-south1-docker.pkg.dev
```

**2. Build (No Cache):**
*Used `--no-cache` to ensure the latest code changes (GenAI region fix) were picked up.*
```bash
docker build --no-cache -t asia-south1-docker.pkg.dev/hackathon-pipeline/governance-app/ai-governance:prod .
```

**3. Push to Artifact Registry:**
```bash
docker push asia-south1-docker.pkg.dev/hackathon-pipeline/governance-app/ai-governance:prod
```

**4. Deploy to Cloud Run:**
```bash
gcloud run services update ai-governance-demo \
   --image asia-south1-docker.pkg.dev/hackathon-pipeline/governance-app/ai-governance:prod \
   --region asia-south1 \
   --memory 2Gi \
   --platform managed \
   --allow-unauthenticated
```

---

## 5. Final Status
*   **URL:** Publicly accessible via Cloud Run.
*   **Data Source:** Live Querying from BigQuery.
*   **State Management:** Persistent via Firestore.
*   **AI:** Operational via Vertex AI (Gemini 2.0 Flash Lite).
*   **Security:** Role-Based Access via Service Account IAM.