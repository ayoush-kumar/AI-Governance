AI Governance - Streamlit Web Application
========================================
This repository contains a Streamlit-based AI Governance dashboard and automation prototype built using Python 3.10.

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
