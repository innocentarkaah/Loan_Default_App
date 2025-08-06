# ===========================================================================
# Loan Default Prediction Web App
# ===========================================================================

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
from xgboost import XGBClassifier

# --------------------------------------------------------------------------
# Page Appearance Setup
# --------------------------------------------------------------------------
sns.set_theme(style='whitegrid', palette='muted')
st.set_page_config(page_title='Loan Default Prediction App', layout='wide')

# --------------------------------------------------------------------------
# Feature Lists & Top-10 Selection
# --------------------------------------------------------------------------
NUMERIC_FEATURES = [
    'Age', 'Income', 'LoanAmount', 'CreditScore',
    'MonthsEmployed', 'NumCreditLines', 'InterestRate',
    'LoanTerm', 'DTIRatio'
]
CATEGORICAL_FEATURES = [
    'Education', 'EmploymentType', 'MaritalStatus',
    'HasMortgage', 'HasDependents', 'LoanPurpose',
    'HasCoSigner'
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TOP_FEATURES = [
    'CreditScore', 'Income', 'Age', 'LoanAmount', 'DTIRatio',
    'MonthsEmployed', 'InterestRate', 'NumCreditLines', 'LoanTerm',
    'Education'
]

# --------------------------------------------------------------------------
# Data Loading and Caching
# --------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str)
    return df

@st.cache_data
def get_class_counts(df: pd.DataFrame):
    return df['Default'].value_counts().sort_index()

# --------------------------------------------------------------------------
# Exploratory Data Analysis
# --------------------------------------------------------------------------
def show_eda(df: pd.DataFrame):
    with st.expander('Exploratory Data Analysis'):
        st.subheader('Summary Statistics')
        st.write(df.describe(include='all'))

        st.subheader('Class Distribution')
        counts = get_class_counts(df)
        fig, ax = plt.subplots()
        sns.barplot(x=counts.index, y=counts.values, ax=ax)
        ax.set_xlabel('Default Flag')
        ax.set_ylabel('Sample Count')
        st.pyplot(fig)

        st.subheader('Correlation Matrix')
        num_cols = df.select_dtypes(include=np.number).columns.drop('Default')
        corr = df[num_cols].corr()
        fig2, ax2 = plt.subplots()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', ax=ax2)
        ax2.set_title('Numeric Feature Correlations')
        st.pyplot(fig2)

        st.subheader('Numeric Feature Histograms')
        for col in num_cols[:6]:
            fig_hist, ax_hist = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax_hist)
            ax_hist.set_title(f'Distribution of {col}')
            st.pyplot(fig_hist)

        st.subheader('Boxplots of Numeric Features')
        melted = df[num_cols].melt(var_name='Feature', value_name='Value')
        fig_box, ax_box = plt.subplots()
        sns.boxplot(x='Feature', y='Value', data=melted, ax=ax_box)
        plt.xticks(rotation=45)
        st.pyplot(fig_box)

# --------------------------------------------------------------------------
# Plot Feature Importances
# --------------------------------------------------------------------------
def plot_feature_importance(model: Pipeline, preprocessor: ColumnTransformer, top_n: int = 10):
    num_feats = preprocessor.transformers_[0][2]
    cat_pipe = preprocessor.transformers_[1][1]
    cat_feats = preprocessor.transformers_[1][2]
    encoded = cat_pipe.named_steps['encoder'].get_feature_names_out(cat_feats)

    names = np.concatenate([num_feats, encoded])
    importances = model.named_steps['classifier'].feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    fig_imp, ax_imp = plt.subplots()
    sns.barplot(x=importances[idx], y=names[idx], orient='h', ax=ax_imp)
    ax_imp.set_title(f'Top {top_n} Features by Importance')
    ax_imp.set_xlabel('Importance Score')
    return fig_imp

# --------------------------------------------------------------------------
# Construct Machine Learning Pipeline
# --------------------------------------------------------------------------
@st.cache_resource
def build_pipeline() -> Pipeline:
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, NUMERIC_FEATURES),
        ('cat', cat_pipe, CATEGORICAL_FEATURES)
    ])
    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    selector = SelectKBest(score_func=f_classif, k='all')
    return Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('classifier', clf)
    ])

# --------------------------------------------------------------------------
# Sidebar Input: only top-10 features + Predict Button
# --------------------------------------------------------------------------
def user_input_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header('Make a Prediction')
    vals = {}
    for feat in TOP_FEATURES:
        if feat in NUMERIC_FEATURES:
            md = float(df[feat].median())
            vals[feat] = st.sidebar.number_input(feat, value=md)
        else:
            opts = df[feat].dropna().unique().tolist()
            vals[feat] = st.sidebar.selectbox(feat, opts)
    inp = pd.DataFrame([vals])
    for feat in ALL_FEATURES:
        if feat not in inp.columns:
            inp[feat] = df[feat].mean() if feat in NUMERIC_FEATURES else df[feat].mode()[0]
    return inp

# --------------------------------------------------------------------------
# Main Application Logic
# --------------------------------------------------------------------------
def main():
    st.title('Loan Default Predictor')

    df = load_data('Loan_default.csv')
    with st.expander('Data Overview', expanded=True):
        st.dataframe(df.head())
        st.dataframe(df.dtypes.to_frame('Type'))
        missing = df.isnull().sum().sum()
        st.write(f'Total missing values: {missing}')

    if 'eda_shown' not in st.session_state:
        show_eda(df)
        st.session_state['eda_shown'] = True

    X = df.drop(columns=['LoanID', 'Default'])
    y = df['Default']

    pipeline = build_pipeline()
    try:
        model = joblib.load('xgb_model.joblib')
        st.sidebar.success('Using cached model')
    except FileNotFoundError:
        st.sidebar.info('Training new model...')
        params = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__subsample': [0.8, 1.0]
        }
        grid = GridSearchCV(pipeline, params, cv=3, scoring='f1', n_jobs=-1)
        model = grid.fit(X, y).best_estimator_
        joblib.dump(model, 'xgb_model.joblib')
        st.sidebar.success(f"Model trained (best params: {grid.best_params_})")

    with st.expander('Model Evaluation', expanded=False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        st.subheader('Performance Metrics')
        rep_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        st.dataframe(rep_df)
        st.subheader('Confusion Matrix')
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax_cm)
        st.pyplot(fig_cm)
        st.subheader('ROC Curve')
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend()
        st.pyplot(fig_roc)
        st.subheader('Key Features (Importance)')
        fig_imp = plot_feature_importance(model, pipeline.named_steps['preprocessor'], top_n=10)
        st.pyplot(fig_imp)

    input_df = user_input_sidebar(df)
    if st.sidebar.button('Predict Default'):
        X_input = input_df[ALL_FEATURES]
        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0][1]
        st.subheader('Prediction Result')
        if pred == 1:
            st.error('🔴 Customer is likely to **Default**')
        else:
            st.success('🟢 Customer is **Not likely to Default**')
        st.subheader('Default Probability')
        st.write(f"**{proba*100:.2f}%**")

if __name__ == '__main__':
    main()
