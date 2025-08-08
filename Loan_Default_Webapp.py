import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import os
import sys

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Title
st.title("Loan Default Prediction App")

# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv("Loan_default.csv")
    return data

df = load_data()

# Show data preview
if st.checkbox("Show raw data"):
    st.write(df.head())

# Sidebar Inputs
st.sidebar.header("Model Configuration")
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
random_state = st.sidebar.number_input("Random State", 0, 100, 42)

# Feature and target
X = df.drop("default", axis=1)
y = df["default"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=test_size, 
                                                    random_state=random_state, 
                                                    stratify=y)

# Preprocessing
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

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

# SMOTE
smote = SMOTE(random_state=random_state)

# Model
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state)

# Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", smote),
    ("classifier", model)
])

# Train Model Button
if st.button("Train Model"):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    fpr, tpr, _ = roc_curve(y_test, pipeline.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    joblib.dump(pipeline, "loan_default_model.pkl")
    st.success("Model trained and saved successfully!")

# Load Saved Model
if st.button("Load Saved Model"):
    if os.path.exists("loan_default_model.pkl"):
        pipeline = joblib.load("loan_default_model.pkl")
        st.success("Model loaded successfully!")
    else:
        st.error("No saved model found.")
