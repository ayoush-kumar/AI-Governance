import pandas as pd
from google.cloud import bigquery
import os

# 1. Configuration
PROJECT_ID = "hackathon-pipeline"
DATASET_ID = "governance_data"
TABLE_ID = "predictions"
CSV_FILE_PATH = "governance_prototype/predictions.csv" # Ensure this path is correct

# 2. Initialize Client
print("Connecting to BigQuery...")
client = bigquery.Client(project=PROJECT_ID)

# 3. Read CSV
print(f"Reading {CSV_FILE_PATH}...")
df = pd.read_csv(CSV_FILE_PATH)

# Fix Date Format (BigQuery needs proper timestamps)
df["created_at"] = pd.to_datetime(df["created_at"])

# 4. Define the Table Address
table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

# 5. Configure Job (Write Truncate = Wipe old data and put new data)
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",
)

# 6. Upload
print(f"Uploading {len(df)} rows to {table_ref}...")
job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)

# 7. Wait for result
job.result()
print("✅ Success! Data uploaded to BigQuery.")