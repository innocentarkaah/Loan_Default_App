# Loan Default Prediction App using Pre-trained Model

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# 1. Load data
# -----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Loan_default.csv")
    df = df.drop(columns=["LoanID"])  # Drop non-predictive ID
    return df

# -----------------------------------
# 2. Load pre-trained model and columns
# -----------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("cat_cols.pkl", "rb") as f:
        cat_cols = pickle.load(f)
    with open("num_cols.pkl", "rb") as f:
        num_cols = pickle.load(f)
    return model, cat_cols, num_cols

# -----------------------------------
# 3. App UI - Data Overview
# -----------------------------------
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

# -----------------------------------
# 4. Streamlit app layout
# -----------------------------------
st.set_page_config(layout="wide")
st.title("🏦 Loan Default Prediction App")

df = load_data()
model, cat_cols, num_cols = load_model()

show_data_overview(df)

# -----------------------------------
# 5. User Input via Sidebar
# -----------------------------------
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

# -----------------------------------
# 6. Make Prediction
# -----------------------------------
with st.expander("📌 Prediction", expanded=True):
    st.subheader("Input Data")
    st.write(input_df)

    if st.sidebar.button("Predict"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        result = "Will Default" if prediction == 1 else "Will Not Default"
        st.success(f"**{result}** with probability: **{probability:.2f}**")

# -----------------------------------
# 7. Optional: Tree Visualization (Top Levels)
# -----------------------------------
with st.expander("🌳 Decision Tree Visualization", expanded=False):
    try:
        from sklearn.tree import plot_tree

        preprocessor = model.named_steps['pre']
        clf = model.named_steps['clf']
        encoded_cat = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
        feat_names = num_cols + list(encoded_cat)

        fig, ax = plt.subplots(figsize=(20, 8))
        plot_tree(
            clf,
            feature_names=feat_names,
            class_names=['No Default', 'Default'],
            filled=True,
            max_depth=3,
            fontsize=10,
            ax=ax
        )
        st.pyplot(fig)
    except Exception as e:
        st.warning("Decision tree visualization not available for this model.")
        st.text(str(e))
