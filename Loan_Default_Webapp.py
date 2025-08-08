# Loan_Default_Webapp.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

# =========================
#  SELF-REBOOT SYSTEM
# =========================
def reboot_app():
    """Clears cache and restarts the Streamlit app."""
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# --- Manual reload button ---
if st.sidebar.button("Reload App"):
    reboot_app()

# --- Automatic reboot if too many session vars ---
max_cache_items = 20  # adjust for your app
if len(st.session_state.keys()) > max_cache_items:
    st.warning("High resource usage detected — rebooting app...")
    reboot_app()

# =========================
#  APP TITLE
# =========================
st.title("Loan Default Prediction App")

# =========================
#  LOAD DATA
# =========================
@st.cache_data
def load_data():
    data = pd.read_csv("Loan_Default_Data.csv")
    return data

try:
    df = load_data()
except FileNotFoundError:
    st.error("Data file not found. Please upload `Loan_Default_Data.csv`.")
    st.stop()

# =========================
#  SIDEBAR USER INPUT
# =========================
st.sidebar.header("Model Configuration")

test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20, step=5) / 100
random_state = st.sidebar.number_input("Random State", value=42, step=1)
use_smote = st.sidebar.checkbox("Use SMOTE for class balancing", value=True)

# =========================
#  FEATURE SELECTION
# =========================
X = df.drop("loan_default", axis=1)
y = df["loan_default"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "category"]).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# =========================
#  MODEL TRAINING
# =========================
@st.cache_resource
def train_model(X, y, test_size, random_state, use_smote):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return model, X_test, y_test, y_pred, y_proba

model, X_test, y_test, y_pred, y_proba = train_model(X, y, test_size, random_state, use_smote)

# =========================
#  MODEL EVALUATION
# =========================
st.subheader("Model Performance")
st.text(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
st.metric("ROC-AUC Score", f"{roc_auc:.3f}")

fig, ax = plt.subplots()
RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
st.pyplot(fig)

# =========================
#  SAVE & LOAD MODEL
# =========================
if st.sidebar.button("Save Model"):
    joblib.dump(model, "loan_default_model.pkl")
    st.sidebar.success("Model saved as `loan_default_model.pkl`")

if st.sidebar.button("Load Model"):
    try:
        model = joblib.load("loan_default_model.pkl")
        st.sidebar.success("Model loaded successfully.")
    except FileNotFoundError:
        st.sidebar.error("Saved model not found.")
