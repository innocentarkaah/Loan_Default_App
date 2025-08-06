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

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

# --------------------------------------------------------------------------
# Page Appearance Setup
# --------------------------------------------------------------------------
# Use a clean Seaborn theme for plots and configure the Streamlit layout
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

# Replace or reorder based on your model's top feature importances
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

# --------------------------------------------------------------------------
# Prepare Class Balance Information
# --------------------------------------------------------------------------
@st.cache_data
def get_balanced_counts(df: pd.DataFrame):
    orig = df['Default'].value_counts().sort_index()
    sm = SMOTE(random_state=42)
    numeric_cols = df.select_dtypes(include=np.number).columns.drop('Default')
    _, resampled_y = sm.fit_resample(df[numeric_cols], df['Default'])
    balanced = pd.Series(resampled_y).value_counts().sort_index()
    return orig, balanced

# --------------------------------------------------------------------------
# Exploratory Data Analysis
# --------------------------------------------------------------------------
def show_eda(df: pd.DataFrame):
    with st.expander('Exploratory Data Analysis'):
        st.subheader('Summary Statistics')
        st.write(df.describe(include='all'))

        st.subheader('Class Distribution')
        orig_counts, balanced_counts = get_balanced_counts(df)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('**Original Data**')
            fig1, ax1 = plt.subplots()
            sns.barplot(x=orig_counts.index, y=orig_counts.values, ax=ax1)
            ax1.set_xlabel('Default Flag')
            ax1.set_ylabel('Sample Count')
            st.pyplot(fig1)
        with col2:
            st.markdown('**After SMOTE Balancing**')
            fig2, ax2 = plt.subplots()
            sns.barplot(x=balanced_counts.index, y=balanced_counts.values, ax=ax2)
            ax2.set_xlabel('Default Flag')
            ax2.set_ylabel('Sample Count')
            st.pyplot(fig2)

        st.subheader('Correlation Matrix')
        num_cols = df.select_dtypes(include=np.number).columns.drop('Default')
        corr = df[num_cols].corr()
        fig3, ax3 = plt.subplots()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', ax=ax3)
        ax3.set_title('Numeric Feature Correlations')
        st.pyplot(fig3)

        st.subheader('Numeric Feature Histograms')
        for col in num_cols[:6]:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f'Distribution of {col}')
            st.pyplot(fig)

        st.subheader('Boxplots of Numeric Features')
        melted = df[num_cols].melt(var_name='Feature', value_name='Value')
        fig4, ax4 = plt.subplots()
        sns.boxplot(x='Feature', y='Value', data=melted, ax=ax4)
        plt.xticks(rotation=45)
        st.pyplot(fig4)

# --------------------------------------------------------------------------
# Plot Feature Importances
# --------------------------------------------------------------------------
def plot_feature_importance(model: ImbPipeline, preprocessor: ColumnTransformer, top_n: int = 10):
    numeric_features = preprocessor.transformers_[0][2]
    cat_pipe = preprocessor.transformers_[1][1]
    categorical_features = preprocessor.transformers_[1][2]
    encoded = cat_pipe.named_steps['encoder'].get_feature_names_out(categorical_features)

    feature_names = np.concatenate([numeric_features, encoded])
    importances = model.named_steps['classifier'].feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    fig_imp, ax_imp = plt.subplots()
    sns.barplot(x=importances[idx], y=feature_names[idx], orient='h', ax=ax_imp)
    ax_imp.set_title(f'Top {top_n} Features by Importance')
    ax_imp.set_xlabel('Importance Score')
    return fig_imp

# --------------------------------------------------------------------------
# Construct Machine Learning Pipeline
# --------------------------------------------------------------------------
@st.cache_resource
def build_pipeline(params=None) -> ImbPipeline:
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
    return ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('selector', selector),
        ('classifier', clf)
    ])

# --------------------------------------------------------------------------
# Sidebar Input: only top-10 features + Predict Button
# --------------------------------------------------------------------------
def user_input_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header('Make a Prediction')
    user_vals = {}
    for feat in TOP_FEATURES:
        if feat in NUMERIC_FEATURES:
            md = float(df[feat].median())
            user_vals[feat] = st.sidebar.number_input(feat, value=md)
        else:
            opts = df[feat].dropna().unique().tolist()
            user_vals[feat] = st.sidebar.selectbox(feat, opts)

    input_df = pd.DataFrame([user_vals])
    # fill remaining features
    for feat in ALL_FEATURES:
        if feat not in input_df.columns:
            if feat in NUMERIC_FEATURES:
                input_df[feat] = df[feat].mean()
            else:
                input_df[feat] = df[feat].mode()[0]
    return input_df

# --------------------------------------------------------------------------
# Main Application Logic
# --------------------------------------------------------------------------
def main():
    st.title('Loan Default Predictor')

    # Load data and show overview
    df = load_data('Loan_default.csv')
    with st.expander('Data Overview', expanded=True):
        st.dataframe(df.head())
        st.dataframe(df.dtypes.to_frame('Type'))
        missing = df.isnull().sum().sum()
        st.write(f'Total missing values: {missing}')

    # Show EDA only once
    if 'eda_shown' not in st.session_state:
        show_eda(df)
        st.session_state['eda_shown'] = True

    # Prepare data
    X = df.drop(columns=['LoanID', 'Default'])
    y = df['Default']

    # Build or load model
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

    # Evaluate section
    with st.expander('Model Evaluation', expanded=False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
        st.subheader('Performance Metrics')
        rep_df = pd.DataFrame(classification_report(y_test,y_pred,output_dict=True)).transpose()
        st.dataframe(rep_df)
        st.subheader('Confusion Matrix')
        cm = confusion_matrix(y_test,y_pred)
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax_cm)
        st.pyplot(fig_cm)
        st.subheader('ROC Curve')
        fpr, tpr, _ = roc_curve(y_test,y_proba)
        roc_auc = auc(fpr,tpr)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr,tpr,label=f'AUC = {roc_auc:.2f}')
        ax_roc.plot([0,1],[0,1],'k--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend()
        st.pyplot(fig_roc)
        st.subheader('Key Features (Importance)')
        fig_imp = plot_feature_importance(model, model.named_steps['preprocessor'], top_n=10)
        st.pyplot(fig_imp)

    # Sidebar input and predict
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

# ===========================================================================
if __name__ == '__main__':
    main()
# ===========================================================================
