import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

# Page configuration: layout and title
st.set_page_config(page_title='Loan Default Prediction App', layout='wide')
sns.set_theme(style='whitegrid', palette='muted')

# Define which features are numerical and which are categorical
NUMERIC_FEATURES = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']

CATEGORICAL_FEATURES = ['Education', 'EmploymentType', 'MaritalStatus',
                        'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Select top features to display in the prediction sidebar
TOP_FEATURES = ['CreditScore', 'Income', 'Age', 'LoanAmount', 'DTIRatio',
                'MonthsEmployed', 'InterestRate', 'NumCreditLines', 'LoanTerm', 'Education']

# Load dataset with Streamlit caching
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str)
    return df

# Calculate original and SMOTE-balanced class distribution
@st.cache_data
def get_balanced_counts(df):
    orig = df['Default'].value_counts().sort_index()
    X = df.select_dtypes(include=np.number).drop(columns='Default')
    y = df['Default']
    sm = SMOTE(random_state=42)
    _, res_y = sm.fit_resample(X, y)
    balanced = pd.Series(res_y).value_counts().sort_index()
    return orig, balanced

# Create a full preprocessing + modeling pipeline
def build_pipeline():
    # Numeric features: impute missing values and scale
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical features: impute and encode using one-hot
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine both pipelines
    preprocessor = ColumnTransformer([
        ('num', num_pipe, NUMERIC_FEATURES),
        ('cat', cat_pipe, CATEGORICAL_FEATURES)
    ])

    # Define the model
    classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    # Full modeling pipeline including SMOTE and feature selection
    return ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('selector', SelectKBest(score_func=f_classif, k='all')),
        ('classifier', classifier)
    ])

# Plot top N important features using model's feature importances
def plot_feature_importance(model, top_n=10):
    preprocessor = model.named_steps['preprocessor']
    num_feats = preprocessor.transformers_[0][2]
    cat_encoder = preprocessor.transformers_[1][1].named_steps['encoder']
    cat_feats = preprocessor.transformers_[1][2]
    encoded_cat_feats = cat_encoder.get_feature_names_out(cat_feats)
    all_feature_names = np.concatenate([num_feats, encoded_cat_feats])
    importances = model.named_steps['classifier'].feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots()
    sns.barplot(x=importances[idx], y=all_feature_names[idx], orient='h', ax=ax)
    ax.set_title('Top Features')
    return fig

# Sidebar input UI for prediction
def user_input_sidebar(df, model_loaded):
    if model_loaded:
        st.sidebar.success("Trained model loaded successfully from file.")

    st.sidebar.header('Make a Prediction')

    data = {}
    for feat in TOP_FEATURES:
        if feat in NUMERIC_FEATURES:
            default = float(df[feat].median())
            data[feat] = st.sidebar.number_input(feat, value=default)
        else:
            options = df[feat].dropna().unique().tolist()
            data[feat] = st.sidebar.selectbox(feat, options)
    inp = pd.DataFrame([data])
    for feat in ALL_FEATURES:
        if feat not in inp.columns:
            inp[feat] = df[feat].mean() if feat in NUMERIC_FEATURES else df[feat].mode()[0]
    return inp

# Main app function
def main():
    st.title('Loan Default Prediction App')
    df = load_data('Loan_default.csv')

    # --- Data Overview section
    with st.expander("🔍 Data Overview", expanded=False):
        overview = st.selectbox(
            "Select what to display:",
            ["Total Missing Values", "Data Head", "Data Types", "Descriptive Statistics"]
        )

        if overview == "Total Missing Values":
            missing = df.isnull().sum().rename("missing_count")
            st.write("Missing values per column:")
            st.dataframe(missing.to_frame())

        elif overview == "Data Head":
            n = st.slider("Number of rows to view:", min_value=5, max_value=50, value=5, step=5)
            st.write(f"First {n} rows of the dataset:")
            st.dataframe(df.head(n))

        elif overview == "Data Types":
            dtypes = df.dtypes.rename("dtype")
            st.write("Column data types:")
            st.dataframe(dtypes.to_frame())

        else:  # Descriptive Statistics
            st.write("Descriptive statistics for numerical columns:")
            st.dataframe(df.describe())

    with st.expander("Exploratory Data Analysis (EDA)", expanded=False):
        st.subheader('Summary Statistics')
        st.write(df.describe(include='all'))

        st.subheader('Class Distribution')
        orig, bal = get_balanced_counts(df)
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots()
            sns.barplot(x=orig.index, y=orig.values, ax=ax1)
            ax1.set_title('Original Data')
            st.pyplot(fig1)
        with col2:
            fig2, ax2 = plt.subplots()
            sns.barplot(x=bal.index, y=bal.values, ax=ax2)
            ax2.set_title('After SMOTE')
            st.pyplot(fig2)

        st.subheader('Correlation Matrix')
        num_cols = df.select_dtypes(include=np.number).columns.drop('Default')
        fig3, ax3 = plt.subplots()
        sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='vlag', ax=ax3)
        st.pyplot(fig3)

        st.subheader('Histograms')
        for i in range(0, 6, 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(num_cols):
                    with cols[j]:
                        fig_hist, ax_hist = plt.subplots()
                        sns.histplot(df[num_cols[i + j]], kde=True, ax=ax_hist)
                        ax_hist.set_title(f'{num_cols[i + j]} Distribution')
                        st.pyplot(fig_hist)

        st.subheader('Boxplot Overview')
        fig_box, ax_box = plt.subplots(figsize=(8, 4))
        melted = df[num_cols].melt(var_name='Feature', value_name='Value')
        sns.boxplot(x='Feature', y='Value', data=melted, ax=ax_box)
        ax_box.set_xticklabels(ax_box.get_xticklabels(), rotation=45)
        st.pyplot(fig_box)

    X = df.drop(columns=['LoanID', 'Default'])
    y = df['Default']
    pipeline = build_pipeline()
    model_loaded = False

    try:
        model = joblib.load('xgb_model.joblib')
        model_loaded = True
    except FileNotFoundError:
        st.info('Training model for the first time...')
        param_grid = {
            'classifier__n_estimators': [100],
            'classifier__max_depth': [3, 5],
            'classifier__learning_rate': [0.01, 0.1]
        }
        grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
        model = grid.fit(X, y).best_estimator_
        joblib.dump(model, 'xgb_model.joblib')
        st.success('Model trained and saved.')

    with st.expander("Model Evaluation", expanded=False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        st.subheader("Classification Report")
        report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        st.dataframe(report)

        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax_cm)
        st.pyplot(fig_cm)

        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_fig, roc_ax = plt.subplots()
        roc_ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
        roc_ax.plot([0, 1], [0, 1], 'k--')
        roc_ax.legend()
        roc_ax.set_xlabel("False Positive Rate")
        roc_ax.set_ylabel("True Positive Rate")
        st.pyplot(roc_fig)

        st.subheader("Feature Importance")
        fig_imp = plot_feature_importance(model)
        st.pyplot(fig_imp)

    with st.expander("Prediction Result", expanded=False):
        inp_df = user_input_sidebar(df, model_loaded=model_loaded)
        if st.sidebar.button("Predict Default"):
            pred = model.predict(inp_df[ALL_FEATURES])[0]
            prob = model.predict_proba(inp_df[ALL_FEATURES])[0][1]
            result = "Likely to Default" if pred else "Not Likely to Default"
            output_df = pd.DataFrame({
                'Prediction': [result],
                'Probability of Default': [f"{prob:.2%}"]
            })
            st.dataframe(output_df)

if __name__ == '__main__':
    main()
