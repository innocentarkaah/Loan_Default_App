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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# -----------------------------------------
# Page Setup: Configure layout and appearance
# -----------------------------------------
sns.set_theme(style="whitegrid", palette="muted")  # Use seaborn theme for consistency
st.set_page_config(page_title="Loan Default Prediction App", layout="wide")
st.title("Loan Default Prediction App")
st.markdown("Supervised Machine Learning Approach")

# -----------------------------------------
# Data Loading: Read and cache dataset
# -----------------------------------------
@st.cache_data
def load_data():
    """
    Load the loan default CSV file into a DataFrame.
    Cached to prevent reloading on each interaction.
    """
    return pd.read_csv("Loan_default.csv")

# -----------------------------------------
# Plot Functions: Cached visualization helpers
# -----------------------------------------
@st.cache_data
def plot_correlation(df):
    """
    Heatmap of correlations for numeric features.
    """
    numeric = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="vlag", linewidths=0.4, ax=ax)
    ax.set_title("Feature Correlation Matrix", fontsize=14)
    plt.tight_layout()
    return fig

@st.cache_data
def plot_target_distribution(df):
    """
    Bar chart showing counts of default vs non-default.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.countplot(x='Default', data=df, palette=['#4c72b0', '#dd8452'], ax=ax)
    ax.set_title("Loan Default Distribution", fontsize=14)
    ax.set_xlabel("Default (0 = No, 1 = Yes)")
    ax.set_ylabel("Count")
    for p in ax.patches:
        ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
    plt.tight_layout()
    return fig

@st.cache_data
def plot_histograms(df, _cols):
    """
    Density histograms with KDE for specified numeric columns.
    Prefix '_' in _cols to skip hashing by Streamlit.
    """
    figures = []
    for col in _cols:
        fig, ax = plt.subplots(figsize=(4, 2))
        sns.histplot(df[col], kde=True, stat='density', edgecolor='white', ax=ax)
        ax.set_title(f"Distribution of {col}", fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        plt.tight_layout()
        figures.append(fig)
    return figures

@st.cache_data
def plot_boxplots(df, _cols):
    """
    Boxplots for key numeric features. '_' used to bypass hashing.
    """
    melt_df = df[_cols].melt(var_name='Feature', value_name='Value')
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='Feature', y='Value', data=melt_df, palette='Set2', ax=ax)
    ax.set_title("Box Plot of Key Numeric Features", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# -----------------------------------------
# Feature Importance: Extract and visualize
# -----------------------------------------
def plot_feature_importance(model, preprocessor, top_n=10):
    raw_names = preprocessor.get_feature_names_out()
    importances = model.named_steps['classifier'].feature_importances_
    clean_names = [name.split('__', 1)[1] if '__' in name else name for name in raw_names]
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [clean_names[i] for i in indices]
    top_importances = importances[indices]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=top_importances, y=top_features, ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    return fig

# -----------------------------------------
# Exploratory Data Analysis (EDA)
# -----------------------------------------
def show_eda(df):
    with st.expander("Exploratory Data Analysis"):
        st.subheader("Summary Statistics")
        st.write(df.describe(include='all'))

        st.subheader("Correlation Matrix")
        st.pyplot(plot_correlation(df))

        st.subheader("Default Distribution")
        st.pyplot(plot_target_distribution(df))

        st.subheader("Histograms for Numeric Features")
        num_cols = df.select_dtypes(include=[np.number]).columns.drop('Default')[:6]
        for fig in plot_histograms(df, num_cols):
            st.pyplot(fig)

        st.subheader("Boxplots for Numeric Features")
        st.pyplot(plot_boxplots(df, num_cols))

# -----------------------------------------
# Pipeline Construction
# -----------------------------------------
def build_pipeline(params=None):
    numeric_features = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                        'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
    categorical_features = ['Education', 'EmploymentType', 'MaritalStatus',
                            'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ])
    xgb_args = params or {'use_label_encoder': False, 'eval_metric': 'logloss'}
    return Pipeline([('preprocessor', preprocessor), ('classifier', XGBClassifier(**xgb_args))])

# -----------------------------------------
# Model Training with Hyperparameter Tuning
# -----------------------------------------
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
    st.sidebar.success(f"Best parameters found: {grid.best_params_}")
    joblib.dump(grid.best_estimator_, 'xgb_model.joblib')
    return grid.best_estimator_

# -----------------------------------------
# Model Evaluation
# -----------------------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    st.subheader('Classification Report')
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
    st.subheader('Confusion Matrix')
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    st.pyplot(fig)

# -----------------------------------------
# User Prediction Interface
# -----------------------------------------
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
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        st.subheader('Prediction Result')
        st.write(f"Will Default? {'Yes' if prediction else 'No'}")
        st.write(f"Default Probability: {probability:.2%}")

# -----------------------------------------
# Main execution flow
# -----------------------------------------
if __name__ == '__main__':
    data = load_data()
    with st.expander('Data Overview'):
        st.subheader('Sample Records and Data Types')
        st.dataframe(data.head())
        st.dataframe(pd.DataFrame(data.dtypes, columns=['Type']))
        st.subheader('Count of Missing Values')
        st.write(data.isnull().sum().sum())
    show_eda(data)
    X = data.drop(['LoanID', 'Default'], axis=1)
    y = data['Default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    try:
        model = joblib.load('xgb_model.joblib')
        st.sidebar.success('Loaded trained model')
    except FileNotFoundError:
        model = train_model(X_train, y_train)
        st.sidebar.success('Trained and saved new model')
    with st.expander('Model Evaluation'):
        evaluate_model(model, X_test, y_test)
    with st.expander('Top Predictors and Importance'):
        preprocessor = model.named_steps['preprocessor']
        st.pyplot(plot_feature_importance(model, preprocessor, top_n=10))
    prediction_interface(model, data)
