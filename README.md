# Customer Retention Analysis using Logistic Regression

**Student:** Tushar Dhanraj Khangare  
**Roll No.:** CM24119  
**Course:** Foundation of Machine Learning (N-PCCCM502T)  
**Institution:** S. B. Jain Institute of Technology, Management & Research, Nagpur  
**Semester:** III/V(B)  
**TAE-1:** Project Based Learning

## Problem Statement
Identify customers who are at risk of leaving a telecom service using a supervised classification model. Logistic Regression is used because it provides a binary decision and an interpretable churn probability.

## Dataset
Public IBM Telco Customer Churn sample dataset:  
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

The data includes demographics, tenure, services, contract/payment information, monthly charges, total charges and the `Churn` target.

## Local execution
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
If the CSV is not present in `data/`, the app downloads it automatically when internet access is available. For offline execution, download the CSV from the source URL and save it as `data/Telco-Customer-Churn.csv`.

## GitHub
```bash
git init
git add .
git commit -m "Initial TAE-1 customer retention project"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/customer-retention-logistic-regression.git
git push -u origin main
```

## Deployment
Recommended: Streamlit Community Cloud. Put the project in a public GitHub repository, open `https://share.streamlit.io/`, create an app, select the repository/branch and `app.py`, then deploy. Copy the generated `streamlit.app` URL into the report.

## Structure
```text
customer-retention-logistic-regression/
├── app.py
├── requirements.txt
├── README.md
├── data/README.md
└── .streamlit/config.toml
```
