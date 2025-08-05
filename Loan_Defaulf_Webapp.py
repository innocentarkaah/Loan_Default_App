import streamlit as st
import pandas as pd
import numpy as np
from math import ceil
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import io

# 1. Data Import and Overview
# ----------------------------
@st.cache_data
def load_data():
    path = "C:/Users/innoc/PycharmProjects/Unsupervised/Text Analytics/Loan_default.csv"
    df = pd.read_csv(path)
    return df

def show_data_overview(df):
    import io
        # Display basic info and first rows in a DataFrame
    st.subheader("Data Info & Head")
    info_df = pd.DataFrame({
        'column': df.columns,
        'non-null count': df.notnull().sum().values,
        'dtype': df.dtypes.astype(str).values
    })
    st.dataframe(info_df)
    st.write(df.head())

    # Missing data overview
    st.subheader("Missing Values by Column")
    missing = df.isnull().sum()
    st.write(missing[missing > 0] if missing.sum() else "No missing values detected.")

    # Dataset summary statistics
    st.subheader("Dataset Summary")
    st.write(df.describe(include='all'))

    # Categorical value counts and count plots (3 per row)
    st.subheader("Categorical Distributions")
    cat_cols = [c for c in df.select_dtypes(include=['object']).columns if c != 'LoanID']
    for i in range(0, len(cat_cols), 3):
        cols = st.columns(3)
        for j, col in enumerate(cat_cols[i:i+3]):
            with cols[j]:
                counts = df[col].value_counts().reset_index()
                counts.columns = [col, 'count']
                st.write(f"**{col}**")
                st.dataframe(counts)
                fig, ax = plt.subplots()
                sns.countplot(data=df, x=col, order=counts[col].tolist(), ax=ax)
                ax.set_title(f"Count Plot of {col}")
                plt.xticks(rotation=45)
                st.pyplot(fig)

    # Histograms (3 per row)
    st.subheader("Histograms of Numerical Features")
    num_cols = list(df.select_dtypes(include=["int64","float64"]).columns.drop("Default"))
    for i in range(0, len(num_cols), 3):
        cols = st.columns(3)
        for j, col in enumerate(num_cols[i:i+3]):
            with cols[j]:
                fig, ax = plt.subplots()
                ax.hist(df[col], bins=30, edgecolor='black')
                ax.set_title(f"Histogram of {col}")
                st.pyplot(fig)

    # Separate Boxplots (3 per row)
    st.subheader("Boxplots of Numerical Features")
    for i in range(0, len(num_cols), 3):
        cols = st.columns(3)
        for j, col in enumerate(num_cols[i:i+3]):
            with cols[j]:
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col], ax=ax)
                ax.set_title(f"Boxplot of {col}")
                st.pyplot(fig)

    # Violin plots of numeric vs Default (3 per row)
    st.subheader("Violin Plots by Default Status")
    for i in range(0, len(num_cols), 3):
        cols = st.columns(3)
        for j, col in enumerate(num_cols[i:i+3]):
            with cols[j]:
                fig, ax = plt.subplots()
                sns.violinplot(x='Default', y=col, data=df, ax=ax)
                ax.set_title(f"Violin Plot of {col} by Default")
                st.pyplot(fig)

    # Pairplot of key numerical features
    st.subheader("Pairplot of Key Numerical Features")
    default_present = 'Default' in df.columns
    sample_numeric = num_cols[:5]
    valid_cols = [c for c in sample_numeric if c in df.columns]
    if default_present:
        valid_cols.append('Default')
    if len(valid_cols) > 1:
        pairplot_df = df[valid_cols]
        fig = sns.pairplot(pairplot_df, hue='Default' if default_present else None, corner=True)
        st.pyplot(fig.fig)

# 2. Model Training with Preprocessing and Tuning Model Training with Preprocessing and Tuning
# ------------------------------------------------
@st.cache_resource
def train_models(df):
    X = df.drop(columns=["LoanID","Default"])
    y = df["Default"]
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()

    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Decision Tree with hyperparameter tuning
    dt_pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('clf', DecisionTreeClassifier(random_state=42))
    ])
    param_grid = {
        'clf__max_depth': [5,10,15,None],
        'clf__min_samples_split': [2,10,20]
    }
    grid_dt = GridSearchCV(dt_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_dt.fit(X_train, y_train)

    # Random Forest for comparison
    rf_pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    rf_pipeline.fit(X_train, y_train)

    # Evaluate
    models = {
        'Decision Tree': grid_dt.best_estimator_,
        'Random Forest': rf_pipeline
    }
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    return grid_dt.best_estimator_, models['Random Forest'], results, cat_cols, num_cols

# 3. Streamlit App Layout
# ------------------------
st.title("Loan Default Prediction App")
df = load_data()
show_data_overview(df)

tree_model, rf_model, results, cat_cols, num_cols = train_models(df)

# 4. User Input for Prediction
# -----------------------------
st.sidebar.header("User Input Parameters")
def user_input():
    data = {}
    for col in num_cols:
        if col in ['DTIRatio','InterestRate']:
            data[col] = st.sidebar.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        else:
            data[col] = st.sidebar.number_input(col, value=float(df[col].mean()))
    for col in cat_cols:
        data[col] = st.sidebar.selectbox(col, df[col].unique())
    return pd.DataFrame([data])

input_df = user_input()
st.subheader("User Input Features")
st.write(input_df)

# 5. Prediction
# -------------
st.subheader("Prediction (Decision Tree)")
pred = tree_model.predict(input_df)[0]
prob = tree_model.predict_proba(input_df)[0][1]
st.write("Default" if pred==1 else "No Default")
st.write(f"Probability: {prob:.2f}")

# 6. Model Evaluation Display
# ---------------------------
st.subheader("Model Evaluation")
# Consolidated performance metrics
metrics_df = pd.DataFrame(
    {model: { 'precision': res['precision'], 'recall': res['recall'], 'f1-score': res['f1']}
     for model, res in results.items()}
).T
st.write("**Performance Metrics**")
st.dataframe(metrics_df.style.format({ 'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}' }))

# Confusion matrices
for name, res in results.items():
    st.markdown(f"### {name} Confusion Matrix")
    cm = res['confusion_matrix']
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap='Blues', alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i,j], ha='center', va='center')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

# 7. Feature Importance & Tree Visualization. Feature Importance & Tree Visualization
# ------------------------------------------
st.subheader("Feature Importances (Decision Tree)")
pre = tree_model.named_steps['preprocess']
onehot_names = pre.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
feat_names = num_cols + list(onehot_names)
importances = tree_model.named_steps['clf'].feature_importances_
idxs = np.argsort(importances)[::-1][:10]
fig, ax = plt.subplots(figsize=(8,6))
ax.barh(
    [feat_names[i] for i in idxs][::-1],
    importances[idxs][::-1]
)
st.pyplot(fig)

st.subheader("Decision Tree Visualization")
fig, ax = plt.subplots(figsize=(20,10))
plot_tree(
    tree_model.named_steps['clf'],
    feature_names=feat_names,
    class_names=['No Default','Default'],
    filled=True,
    max_depth=3,
    fontsize=10,
    ax=ax
)
st.pyplot(fig)

# 8. Generate PDF Report
# -----------------------
from matplotlib.backends.backend_pdf import PdfPages
import io

if st.button("Download PDF Report"):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        # Page 1: Summary table
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        summary = df.describe().round(2)
        table = ax.table(
            cellText=summary.values,
            colLabels=summary.columns,
            rowLabels=summary.index,
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Feature importances
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh([
            feat_names[i] for i in idxs
        ][::-1], importances[idxs][::-1])
        ax.set_title('Top 10 Feature Importances')
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: Decision tree plot (top levels)
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(
            tree_model.named_steps['clf'],
            feature_names=feat_names,
            class_names=['No Default','Default'],
            filled=True,
            max_depth=3,
            fontsize=8,
            ax=ax
        )
        pdf.savefig(fig)
        plt.close(fig)

    buffer.seek(0)
    st.download_button(
        label="Download Report as PDF",
        data=buffer.getvalue(),
        file_name='loan_default_report.pdf',
        mime='application/pdf'
    )
