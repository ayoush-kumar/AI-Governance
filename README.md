AI Governance - Streamlit Web Application
========================================
This repository contains a Streamlit-based AI Governance dashboard and automation prototype built using Python 3.10.

## Overview
The **AI-Powered Governance Platform** leverages predictive analytics and real-time insights to proactively manage citizen service requests, predict potential service disruptions, and optimize resource allocation across various public departments. The platform integrates real-time data, AI models, and dynamic prioritization to enable government officials to address issues before they escalate, improving the efficiency and responsiveness of public services.

## Project Architecture
<img width="2282" height="1592" alt="diagram" src="https://github.com/user-attachments/assets/7d15c167-6409-4136-bc18-38a90d5dae91" />

## Key Features
- **Predictive Analytics**: AI models forecast SLA breaches, identify service bottlenecks, and assess resource availability.
- **Real-Time Dashboard**: A Streamlit-based dashboard for visualizing key metrics, SLA breaches, risk analysis, and ticket priority scores.
- **Synthetic and Real-Time Data**: Initially uses synthetic data for testing and transitions to real-time data via Google Cloud Pub/Sub for continuous updates.
- **Proactive Service Management**: Prioritizes service requests based on predicted risk and resource allocation.
- **Data-Driven Decision Making**: Provides actionable insights to help officials make informed decisions and allocate resources effectively.

## Target Audience
- **Government Officials & Decision Makers**: Department heads, ministry officials, and managers responsible for public service delivery.
- **Operations & Resource Managers**: Focused on resource allocation, service monitoring, and bottleneck resolution.
- **Public Service Managers**: Aimed at improving public service efficiency across various departments.

## Architecture

### Main Components
1. **Data Ingestion**: Real-time service data is streamed from ministry databases using **Google Cloud Pub/Sub**.
2. **Data Storage**: **Google BigQuery** serves as the centralized data warehouse for storing raw, processed, and prediction data.
3. **Model Training & Predictions**: The AI model predicts SLA breaches, ticket priority, and risk using historical data and real-time inputs.
4. **Streamlit Dashboard**: The frontend dashboard is built using **Streamlit**, which displays key metrics, predictions, and visualizations in real time.
5. **Cloud Run**: The Streamlit app is containerized and deployed using **Google Cloud Run** for serverless, scalable hosting.
6. **Security**: **IAM**, **VPC**, and **KMS** ensure data security, compliance, and access control.

### Data Flow
1. **Synthetic Data** (for testing) is ingested into the platform via **Pub/Sub**.
2. The data is stored and processed in **BigQuery**.
3. The AI model is trained on this data to predict SLA breaches and other service metrics.
4. **Streamlit** pulls the data from **BigQuery** and visualizes key insights for decision-makers.
5. Real-time data from ministry sources will be integrated into the system through **Pub/Sub** as the platform evolves.

## Google Cloud Tools Usage
- **Vertex AI**: Used for training, deployment, and management of AI models.
- **BigQuery**: Serves as the data warehouse for processing and querying service request data.
- **Pub/Sub**: Streams real-time data from external sources into the platform.
- **Cloud Run**: Hosts the **Streamlit** application for real-time, scalable web interfaces.
- **IAM and VPC**: Provides secure access control and private networking.
- **KMS**: Ensures data encryption for compliance with privacy standards.

## How It Works
1. **Data Collection**: Data is ingested into the system from government ministry sources (initially synthetic data and later real-time data from **Pub/Sub**).
2. **Model Prediction**: The platform uses a trained **AI model** to predict the likelihood of SLA breaches and identify high-priority service requests.
3. **Dynamic Prioritization**: Service requests are dynamically prioritized based on predicted risks and resource availability.
4. **Decision-Making**: Users access the **Streamlit dashboard** to view real-time insights and take proactive actions such as reallocating resources or addressing high-priority issues.

Requirements
------------
- Python 3.10
- pip (Python package manager)
- Git (optional)

Installation & Setup
--------------------
1. Clone the Repository:
   git clone https://github.com/ayoush-kumar/AI-Governance.git
   cd AI-Governance

2. Create a Virtual Environment (Python 3.10)

   Mac / Linux:
      python3.10 -m venv venv
      source venv/bin/activate

   Windows:
      py -3.10 -m venv venv
      venv\Scripts\activate

3. Install Dependencies:
   pip install --upgrade pip
   pip install -r requirements.txt

Running the Application
-----------------------
Start the Streamlit app:
   streamlit run main.py

*If your entry file changes (e.g., dashboard_streamlit.py), replace main.py accordingly.*

## Project Structure
```
AI-Governance/
├── Dashboard/
│   └── dashboard_streamlit.py
├── governance_prototype/
│   ├── feature_importance.csv
│   ├── predictions.csv
│   ├── sla_breach_model.joblib
│   └── synthetic_tickets.csv
├── module/
│   └── hackathonpipeline.py
├── main.py
├── requirements.txt
└── README.md
```

Virtual Environment Commands
----------------------------
Deactivate environment:
   deactivate

Update requirements after installing new packages:
   pip freeze > requirements.txt

Contribution
------------
Feel free to submit issues or pull requests. Improvements & feedback are welcome!


