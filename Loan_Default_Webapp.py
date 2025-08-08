import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gc
import psutil

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ====================================================
# SELF-REBOOT SYSTEM
# ====================================================

def reboot_app():
    """Clears cache and restarts the Streamlit app."""
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# --- Memory threshold ---
MEMORY_THRESHOLD = 85  # % RAM usage before reboot
memory_usage = psutil.virtual_memory().percent

if memory_usage > MEMORY_THRESHOLD:
    st.warning(f" High memory usage detected ({memory_usage}%). Rebooting app...")
    reboot_app()

# ====================================================
# CACHED FUNCTIONS
# ====================================================

@st.cache_data
def load_data(path):
    """Load dataset from CSV."""
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    """Load pre-trained ML model."""
    return joblib.load(path)

# ====================================================
# MAIN APP
# ====================================================
def main():
    st.set_page_config(page_title="Loan Default Prediction", layout="wide")
    st.title("💳 Loan Default Prediction App")

    # ======== PULSING RELOAD BUTTON IF MEMORY HIGH ========
    reload_style = """
        <style>
        @keyframes pulse {
            0% { box-shadow: 0 0 5px red; }
            50% { box-shadow: 0 0 20px red; }
            100% { box-shadow: 0 0 5px red; }
        }
        div[data-testid="stButton"] > button.red-reload {
            background-color: red;
            color: white;
            font-weight: bold;
            animation: pulse 1s infinite;
            border: none;
            border-radius: 8px;
        }
        </style>
    """
    st.markdown(reload_style, unsafe_allow_html=True)

    if memory_usage > 75:  # show red pulsing reload if memory high
        if st.button(" Reload App (High Memory!)", key="reload_high", help="Click to free memory", type="secondary"):
            reboot_app()
        st.markdown(
            "<script>document.querySelector('button[kind=secondary]').classList.add('red-reload')</script>",
            unsafe_allow_html=True
        )
    else:
        if st.button(" Reload App", key="reload_normal", help="Click to refresh the app"):
            reboot_app()

    # Load dataset
    try:
        df = load_data("Loan_Default_Data.csv")
    except FileNotFoundError:
        st.error(" Dataset file not found. Please upload Loan_Default_Data.csv")
        return

    st.subheader(" Dataset Preview")
    st.write(df.head())

    # Sidebar for user inputs
    st.sidebar.header("User Input Features")
    user_inputs = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            user_inputs[col] = st.sidebar.selectbox(col, df[col].unique())
        else:
            user_inputs[col] = st.sidebar.number_input(
                col, 
                float(df[col].min()), 
                float(df[col].max()), 
                float(df[col].mean())
            )
    input_df = pd.DataFrame([user_inputs])

    # Load model or train if not available
    try:
        model = load_model("xgb_model.pkl")
    except FileNotFoundError:
        st.warning(" Model file not found. Training new model...")
        # Prepare data
        X = df.drop("Loan_Default", axis=1)
        y = df["Loan_Default"]

        # Preprocessing
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Model pipeline
        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE()),
            ('classifier', XGBClassifier(eval_metric='logloss', use_label_encoder=False))
        ])

        clf.fit(X, y)
        joblib.dump(clf, "xgb_model.pkl")
        model = clf

    # Predictions
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader(" Prediction")
    st.write("Default" if prediction[0] == 1 else "No Default")

    st.subheader("Prediction Probability")
    st.write(prediction_proba)

    # Evaluate model performance
    X = df.drop("Loan_Default", axis=1)
    y = df["Loan_Default"]
    y_pred = model.predict(X)

    st.subheader("Model Evaluation")
    st.text(classification_report(y, y_pred))

    # Free memory
    del df, input_df, X, y, y_pred
    gc.collect()

if __name__ == "__main__":
    main()
