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
# Page Configuration
# --------------------------------------------------------------------------
sns.set_theme(style='whitegrid', palette='muted')
st.set_page_config(page_title='Loan Default Prediction App', layout='wide')

# --------------------------------------------------------------------------
# Load and Cache Data
# --------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV and convert object columns to string.
    """
    df = pd.read_csv(path)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str)
    return df

# --------------------------------------------------------------------------
# Exploratory Data Analysis (EDA)
# --------------------------------------------------------------------------
def show_eda(df: pd.DataFrame):
    with st.expander('Exploratory Data Analysis'):
        # Summary statistics
        st.subheader('Summary Statistics')
        st.write(df.describe(include='all'))

                # Class distribution (original vs SMOTE-balanced side by side)
        st.subheader('Class Distribution')
        orig_counts = df['Default'].value_counts().sort_index()
        # Compute SMOTE-balanced distribution
        sm = SMOTE(random_state=42)
        num_cols = df.select_dtypes(include=np.number).columns.drop('Default')
        X_sm = df[num_cols]
        y_sm = df['Default']
        _, y_res = sm.fit_resample(X_sm, y_sm)
        bal_counts = pd.Series(y_res).value_counts().sort_index()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('**Original Distribution**')
            fig1, ax1 = plt.subplots(figsize=(4,3))
            sns.barplot(x=orig_counts.index, y=orig_counts.values, palette=['#4c72b0','#dd8452'], ax=ax1)
            ax1.set_xlabel('Default')
            ax1.set_ylabel('Count')
            st.pyplot(fig1)
        with col2:
            st.markdown('**SMOTE-Balanced Distribution**')
            fig2, ax2 = plt.subplots(figsize=(4,3))
            sns.barplot(x=bal_counts.index, y=bal_counts.values, palette=['#4c72b0','#dd8452'], ax=ax2)
            ax2.set_xlabel('Default')
            ax2.set_ylabel('Count')
            st.pyplot(fig2)

        # Correlation matrix
        st.subheader('Correlation Matrix')
        numeric_cols = df.select_dtypes(include=np.number).columns.drop('Default')
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(3, 2))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', linewidths=0.4, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)

        # Histograms
        st.subheader('Histograms of Numeric Features')
        for col in numeric_cols[:6]:
            fig, ax = plt.subplots(figsize=(3, 1.5))
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f'Distribution of {col}')
            st.pyplot(fig)

        # Boxplots
        st.subheader('Boxplots of Numeric Features')
        melt = df[numeric_cols].melt(var_name='Feature', value_name='Value')
        fig, ax = plt.subplots(figsize=(3, 2))
        sns.boxplot(x='Feature', y='Value', data=melt, palette='Set2', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

# --------------------------------------------------------------------------
# Feature Importance Visualization
# --------------------------------------------------------------------------
def plot_feature_importance(model: Pipeline, preprocessor: ColumnTransformer, top_n: int = 10):
    """
    Extract and plot the top_n feature importances from the trained XGBoost model with improved readability.
    """
    # Retrieve feature names after preprocessing
    numeric_feats = preprocessor.transformers_[0][2]
    cat_transformer = preprocessor.transformers_[1][1]
    cat_feats = preprocessor.transformers_[1][2]
    encoded_cat = cat_transformer.named_steps['encoder'].get_feature_names_out(cat_feats)
    feature_names = np.concatenate([numeric_feats, encoded_cat])

    # Extract feature importances
    importances = model.named_steps['classifier'].feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = feature_names[indices]
    top_importances = importances[indices]

    # Plot horizontal bar chart with smaller labels
    fig, ax = plt.subplots(figsize=(4, 2.5))
    sns.barplot(x=top_importances, y=top_features, ax=ax, orient='h')
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=8)
    ax.set_ylabel('Feature', fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=7)
    plt.tight_layout()
    return fig

# --------------------------------------------------------------------------
# Build preprocessing and modeling pipeline
# --------------------------------------------------------------------------
def build_pipeline(params=None) -> Pipeline:
    numeric_features = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                        'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
    categorical_features = ['Education', 'EmploymentType', 'MaritalStatus',
                            'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

    # Numeric transformer
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    # Categorical transformer
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ])

    xgb_args = params or {'use_label_encoder': False, 'eval_metric': 'logloss'}
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(**xgb_args))
    ])
    return pipeline

# --------------------------------------------------------------------------
# Train model with hyperparameter tuning
# --------------------------------------------------------------------------
def train_model(X, y):
    pipeline = build_pipeline()
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__subsample': [0.8, 1.0]
    }
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X, y)
    st.sidebar.success(f"Best parameters: {grid.best_params_}")
    joblib.dump(grid.best_estimator_, 'xgb_model.joblib')
    return grid.best_estimator_

# --------------------------------------------------------------------------
# Evaluate model and display results
# --------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    st.subheader('Classification Report')
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    st.dataframe(report_df)

    st.subheader('Confusion Matrix')
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(3, 2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    st.subheader('ROC Curve')
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    st.pyplot(fig)

# --------------------------------------------------------------------------
# Prediction interface in sidebar
# --------------------------------------------------------------------------
def prediction_interface(model, df):
    st.sidebar.header('Make a Prediction')
    numeric_feats = ['Age', 'Income', 'LoanAmount', 'CreditScore',
                     'MonthsEmployed', 'NumCreditLines', 'InterestRate',
                     'LoanTerm', 'DTIRatio']
    user_input = {feat: st.sidebar.number_input(feat, value=float(df[feat].median())) for feat in numeric_feats}
    cat_feats = ['Education', 'EmploymentType', 'MaritalStatus',
                 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
    for feat in cat_feats:
        user_input[feat] = st.sidebar.selectbox(feat, df[feat].unique())

    if st.sidebar.button('Predict Default'):
        input_df = pd.DataFrame([user_input])
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        st.subheader('Prediction Result')
        st.write(f"Will Default? {'Yes' if pred else 'No'}")
        st.write(f"Default Probability: {proba:.2%}")

# --------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------
if __name__ == '__main__':
    df = load_data('Loan_default.csv')
    # Display data overview: head, dtypes, total missing values
    with st.expander('Data Overview', expanded=True):
        st.subheader('Data Snapshot (Head)')
        st.dataframe(df.head())
        st.subheader('Data Types')
        st.dataframe(pd.DataFrame(df.dtypes, columns=['Type']))
        st.subheader('Total Missing Values')
        st.write(df.isnull().sum().sum())

    # Perform exploratory data analysis
    show_eda(df)
    X = df.drop(columns=['LoanID', 'Default'])
    y = df['Default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        model = joblib.load('xgb_model.joblib')
        st.sidebar.success('Loaded trained model')
    except FileNotFoundError:
        model = train_model(X_train, y_train)
        st.sidebar.success('Trained and saved model')

    evaluate_model(model, X_test, y_test)
    # Plot feature importances
    fig_imp = plot_feature_importance(model, model.named_steps['preprocessor'], top_n=10)
    st.pyplot(fig_imp)
    prediction_interface(model, df)

# ===========================================================================
