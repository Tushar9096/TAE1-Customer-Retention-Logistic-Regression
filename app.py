import io
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Customer Retention Analysis", page_icon="📊", layout="wide")
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
LOCAL_DATA = Path("data/Telco-Customer-Churn.csv")

@st.cache_data(show_spinner=False)
def load_data():
    if LOCAL_DATA.exists():
        return pd.read_csv(LOCAL_DATA), "Local IBM Telco Customer Churn CSV"
    with urllib.request.urlopen(DATA_URL, timeout=20) as response:
        return pd.read_csv(io.BytesIO(response.read())), "IBM Telco Customer Churn dataset (downloaded at runtime)"

@st.cache_resource(show_spinner=False)
def train_model(df):
    data=df.copy()
    data["TotalCharges"]=pd.to_numeric(data["TotalCharges"], errors="coerce")
    data=data.dropna(subset=["TotalCharges"]).copy()
    data["Churn"]=(data["Churn"]=="Yes").astype(int)
    X=data.drop(columns=["customerID","Churn"]); y=data["Churn"]
    nums=["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]
    cats=[c for c in X.columns if c not in nums]
    prep=ColumnTransformer([("num",StandardScaler(),nums),("cat",OneHotEncoder(handle_unknown="ignore"),cats)])
    pipe=Pipeline([("preprocessor",prep),("model",LogisticRegression(max_iter=1500,class_weight="balanced",solver="liblinear",random_state=42))])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
    pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); prob=pipe.predict_proba(Xte)[:,1]
    metrics={"Accuracy":accuracy_score(yte,pred),"Precision":precision_score(yte,pred,zero_division=0),"Recall":recall_score(yte,pred,zero_division=0),"F1 Score":f1_score(yte,pred,zero_division=0),"ROC-AUC":roc_auc_score(yte,prob)}
    return pipe,data,yte,pred,metrics,confusion_matrix(yte,pred),classification_report(yte,pred,target_names=["Retained","Churned"],output_dict=True)

def risk_band(p):
    return ("High Risk","Prioritize immediate retention action.") if p>=.70 else (("Medium Risk","Consider a targeted retention offer or service follow-up.") if p>=.40 else ("Low Risk","No urgent retention action indicated by this model."))

st.title("📊 Customer Retention Analysis using Logistic Regression")
st.caption("TAE-1 Project Based Learning | Foundation of Machine Learning | CM24119 – Tushar Dhanraj Khangare")
try:
    df,source=load_data()
    model,data,yte,pred,metrics,cm,report=train_model(df)
except Exception as e:
    st.error("Dataset could not be loaded. Place the IBM CSV at data/Telco-Customer-Churn.csv or enable internet access.")
    st.exception(e); st.stop()

with st.sidebar:
    st.header("Project Information")
    st.write("**Problem:** Identify customers at risk of leaving a telecom service.")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Dataset:** IBM Telco Customer Churn")
    st.write(f"**Records used:** {len(data):,}")
    st.write("**Test split:** 20% stratified")
    st.write("**Class balancing:** balanced class weights")
    st.divider()
    st.info("Academic decision-support prototype: a high-risk score should trigger human review, not automatic customer action.")

t1,t2,t3=st.tabs(["📈 Model Performance","🔎 Churn Analysis","🎯 Customer Risk Predictor"])
with t1:
    st.subheader("Evaluation Metrics")
    cols=st.columns(5)
    for c,(n,v) in zip(cols,metrics.items()): c.metric(n,f"{v:.3f}")
    a,b=st.columns(2)
    with a:
        st.subheader("Confusion Matrix")
        st.dataframe(pd.DataFrame(cm,index=["Actual Retained","Actual Churned"],columns=["Predicted Retained","Predicted Churned"]),use_container_width=True)
    with b:
        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(report).T.round(3),use_container_width=True)
    st.write("Recall is important for retention because false negatives represent customers likely to churn who were not flagged. Precision helps limit unnecessary interventions.")

with t2:
    st.subheader("Dataset Overview")
    a,b,c=st.columns(3)
    a.metric("Customers",f"{len(data):,}"); b.metric("Churned",f"{data.Churn.sum():,}"); c.metric("Churn Rate",f"{data.Churn.mean()*100:.1f}%")
    for title,col in [("Churn by Contract Type","Contract"),("Churn by Internet Service","InternetService")]:
        st.subheader(title)
        tab=data.groupby(col)["Churn"].agg(["count","sum","mean"]).rename(columns={"count":"Customers","sum":"Churned","mean":"Churn Rate"})
        tab["Churn Rate"]=(tab["Churn Rate"]*100).round(2)
        st.dataframe(tab,use_container_width=True)
    st.subheader("Modeling Pipeline")
    st.code("Raw Data → Cleaning → Stratified Train/Test Split → Encoding + Scaling → Logistic Regression → Probability Prediction → Risk Band → Retention Action")

with t3:
    st.subheader("Predict Individual Customer Churn Risk")
    c1,c2,c3=st.columns(3)
    with c1:
        gender=st.selectbox("Gender",["Female","Male"]); senior=st.selectbox("Senior Citizen",[0,1],format_func=lambda x:"Yes" if x else "No")
        partner=st.selectbox("Partner",["Yes","No"]); dependents=st.selectbox("Dependents",["Yes","No"])
        tenure=st.slider("Tenure (months)",0,72,12); phone=st.selectbox("Phone Service",["Yes","No"])
    with c2:
        multiple=st.selectbox("Multiple Lines",["Yes","No","No phone service"]); internet=st.selectbox("Internet Service",["DSL","Fiber optic","No"])
        security=st.selectbox("Online Security",["Yes","No","No internet service"]); backup=st.selectbox("Online Backup",["Yes","No","No internet service"])
        device=st.selectbox("Device Protection",["Yes","No","No internet service"]); tech=st.selectbox("Tech Support",["Yes","No","No internet service"])
    with c3:
        tv=st.selectbox("Streaming TV",["Yes","No","No internet service"]); movies=st.selectbox("Streaming Movies",["Yes","No","No internet service"])
        contract=st.selectbox("Contract",["Month-to-month","One year","Two year"]); paperless=st.selectbox("Paperless Billing",["Yes","No"])
        payment=st.selectbox("Payment Method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
        monthly=st.number_input("Monthly Charges",18.0,120.0,70.0,.01); total=st.number_input("Total Charges",0.0,9000.0,float(monthly*max(tenure,1)),.01)
    row=pd.DataFrame([{"gender":gender,"SeniorCitizen":senior,"Partner":partner,"Dependents":dependents,"tenure":tenure,"PhoneService":phone,"MultipleLines":multiple,"InternetService":internet,"OnlineSecurity":security,"OnlineBackup":backup,"DeviceProtection":device,"TechSupport":tech,"StreamingTV":tv,"StreamingMovies":movies,"Contract":contract,"PaperlessBilling":paperless,"PaymentMethod":payment,"MonthlyCharges":monthly,"TotalCharges":total}])
    if st.button("🔮 Predict Risk",type="primary",use_container_width=True):
        p=float(model.predict_proba(row)[:,1][0]); label,advice=risk_band(p)
        st.metric("Estimated Churn Probability",f"{p*100:.1f}%")
        (st.error if label=="High Risk" else st.warning if label=="Medium Risk" else st.success)(f"**{label}** — {advice}")
        st.write("### Suggested retention approach")
        if p>=.70: st.write("- Contact promptly.\n- Review contract, billing and service experience.\n- Consider a targeted retention offer.")
        elif p>=.40: st.write("- Monitor the account.\n- Offer relevant service/support options.\n- Follow up before renewal.")
        else: st.write("- Continue normal engagement.\n- Monitor future changes in account behavior.")
st.divider()
st.caption(f"Data source: {source}. Model: scikit-learn LogisticRegression.")
