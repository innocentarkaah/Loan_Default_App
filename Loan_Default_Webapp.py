# Loan Default Prediction App with Hideable Sections and Comments

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1. Load and preprocess data
# ---------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("Loan_default.csv")
    df = df.drop(columns=["LoanID"])  # Drop LoanID as it's non-predictive and causes serialization issues
    return df

# ----------------------------
# 2. Data overview UI section
# ----------------------------

def show_data_overview(df):
    with st.expander("📊 Data Overview", expanded=False):

        st.subheader("Data Info")
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Non-Null Count': df.notnull().sum().values,
            'Dtype': df.dtypes.astype(str).values
        })
        st.dataframe(info_df)

        st.subheader("Sample Rows")
        st.write(df.head())

        st.subheader("Missing Values")
        missing = df.isnull().sum()
        st.write(missing[missing > 0] if missing.sum() else "✅ No missing values.")

        st.subheader("Summary Statistics")
        st.write(df.describe(include='all'))

        # Categorical Columns
        st.subheader("Categorical Columns Overview")
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        for i in range(0, len(cat_cols), 3):
            cols = st.columns(3)
            for j, col in enumerate(cat_cols[i:i+3]):
                with cols[j]:
                    st.write(f"**{col}**")
                    st.dataframe(df[col].value_counts())
                    fig, ax = plt.subplots()
                    sns.countplot(data=df, x=col, ax=ax)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

        # Numerical Columns
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('Default')

        st.subheader("Histograms")
        for i in range(0, len(num_cols), 3):
            cols = st.columns(3)
            for j, col in enumerate(num_cols[i:i+3]):
                with cols[j]:
                    fig, ax = plt.subplots()
                    ax.hist(df[col], bins=30, edgecolor='black')
                    ax.set_title(f"{col}")
                    st.pyplot(fig)

        st.subheader("Boxplots")
        for i in range(0, len(num_cols), 3):
            cols = st.columns(3)
            for j, col in enumerate(num_cols[i:i+3]):
                with cols[j]:
                    fig, ax = plt.subplots()
                    sns.boxplot(x=df[col], ax=ax)
                    ax.set_title(f"{col}")
                    st.pyplot(fig)

        st.subheader("Violin Plots by Default")
        for i in range(0, len(num_cols), 3):
            cols = st.columns(3)
            for j, col in enumerate(num_cols[i:i+3]):
                with cols[j]:
                    fig, ax = plt.subplots()
                    sns.violinplot(x='Default', y=col, data=df, ax=ax)
                    ax.set_title(f"{col}")
                    st.pyplot(fig)

# -----------------------------------------
# 3. Train and evaluate ML models (cached)
# -----------------------------------------

@st.cache_resource
def train_models(df):
    X = df.drop(columns=['Default'])  # Input features
    y = df['Default']                 # Target variable

    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Preprocessor to scale numeric and encode categorical data
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    # Decision Tree with GridSearch
    dt_pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', DecisionTreeClassifier(random_state=42))
    ])
    param_grid = {
        'clf__max_depth': [5, 10, 15, None],
        'clf__min_samples_split': [2, 10, 20]
    }
    grid = GridSearchCV(dt_pipe, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)

    # Random Forest
    rf_pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    rf_pipe.fit(X_train, y_train)

    models = {
        'Decision Tree': grid.best_estimator_,
        'Random Forest': rf_pipe
    }

    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'cm': confusion_matrix(y_test, y_pred)
        }

    return grid.best_estimator_, rf_pipe, results, cat_cols, num_cols

# ----------------------------------
# 4. Streamlit page + user input UI
# ----------------------------------

st.set_page_config(layout="wide")
st.title("🏦 Loan Default Prediction App")

df = load_data()
show_data_overview(df)
tree_model, rf_model, results, cat_cols, num_cols = train_models(df)

# Sidebar input for prediction
st.sidebar.header("📥 Applicant Features")

def user_input():
    data = {}
    for col in num_cols:
        if col in ['DTIRatio', 'InterestRate']:
            data[col] = st.sidebar.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        else:
            data[col] = st.sidebar.number_input(col, value=float(df[col].mean()))
    for col in cat_cols:
        data[col] = st.sidebar.selectbox(col, df[col].unique())
    return pd.DataFrame([data])

input_df = user_input()

# -----------------------
# 5. Show Input and Predict
# -----------------------

with st.expander("📌 Prediction", expanded=True):
    st.subheader("Input Data")
    st.write(input_df)

    if st.sidebar.button("Predict"):
        pred = tree_model.predict(input_df)[0]
        prob = tree_model.predict_proba(input_df)[0][1]
        result = "Will Default" if pred == 1 else "Will Not Default"
        st.success(f"**{result}** with probability: **{prob:.2f}**")

# -----------------------
# 6. Show Performance
# -----------------------

with st.expander("📈 Model Performance", expanded=False):
    st.subheader("Metrics")
    metric_df = pd.DataFrame({
        model: {
            'precision': res['precision'],
            'recall': res['recall'],
            'f1-score': res['f1']
        }
        for model, res in results.items()
    }).T
    st.dataframe(metric_df.style.format("{:.2f}"))

    for model, res in results.items():
        st.write(f"**{model} - Confusion Matrix**")
        cm = res['cm']
        fig, ax = plt.subplots()
        ax.matshow(cm, cmap='Blues', alpha=0.3)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], va='center', ha='center')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

# -----------------------
# 7. Feature Importance
# -----------------------

with st.expander("📊 Feature Importance", expanded=False):
    st.subheader("Top 10 Features")
    pre = tree_model.named_steps['pre']
    encoded_cat = pre.named_transformers_['cat'].get_feature_names_out(cat_cols)
    feat_names = num_cols + list(encoded_cat)
    importances = tree_model.named_steps['clf'].feature_importances_
    idx = np.argsort(importances)[::-1][:10]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feat_names[i] for i in idx][::-1], importances[idx][::-1])
    ax.set_title("Top 10 Feature Importances")
    st.pyplot(fig)

# -----------------------
# 8. Tree Visualization
# -----------------------

with st.expander("🌳 Decision Tree (Top Levels)", expanded=False):
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(
        tree_model.named_steps['clf'],
        feature_names=feat_names,
        class_names=['No Default', 'Default'],
        filled=True,
        max_depth=3,
        fontsize=10,
        ax=ax
    )
    st.pyplot(fig)
