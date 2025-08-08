import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

# Page configuration
st.set_page_config(page_title='Loan Default Prediction App', layout='wide')
sns.set_theme(style='whitegrid', palette='muted')

# Define features
NUMERIC_FEATURES = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
CATEGORICAL_FEATURES = ['Education', 'EmploymentType', 'MaritalStatus',
                        'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Load dataset with caching
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str)
    return df

# Optimized class distribution calculation
@st.cache_data
def get_balanced_counts(df):
    orig = df['Default'].value_counts().sort_index()
    majority_count = orig.max()
    balanced = pd.Series([majority_count, majority_count], index=orig.index)
    return orig, balanced

# Build full pipeline
def build_pipeline():
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, NUMERIC_FEATURES),
        ('cat', cat_pipe, CATEGORICAL_FEATURES)
    ])
    classifier = XGBClassifier(eval_metric='logloss', random_state=42)
    return ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('selector', SelectKBest(score_func=f_classif, k='all')),
        ('classifier', classifier)
    ])

# Plot feature importances
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
    plt.close(fig)  # Prevents duplicate display
    return fig

# Sidebar for user input
def user_input_sidebar(df, model_loaded):
    if model_loaded:
        st.sidebar.success("Trained model loaded successfully from file.")
    st.sidebar.header('Make a Prediction')
    data = {}
    for feat in ALL_FEATURES:
        if feat in NUMERIC_FEATURES:
            default = float(df[feat].median())
            data[feat] = st.sidebar.number_input(
                feat,
                value=default,
                step=1.0 if np.issubdtype(df[feat].dtype, np.integer) else 0.01
            )
        else:
            options = sorted(df[feat].dropna().unique().tolist())
            default_opt = df[feat].mode()[0]
            data[feat] = st.sidebar.selectbox(feat, options, index=options.index(default_opt))
    return pd.DataFrame([data])

# Cache model training/loading
@st.cache_resource
def get_model():
    df = load_data('Loan_default.csv')
    X = df.drop(columns=['LoanID', 'Default'])
    y = df['Default']
    try:
        return joblib.load('xgb_model.joblib'), True
    except FileNotFoundError:
        pipeline = build_pipeline()
        param_grid = {
            'classifier__n_estimators': [100],
            'classifier__max_depth': [3, 5],
            'classifier__learning_rate': [0.01, 0.1]
        }
        grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=1)  # Reduced parallelism
        model = grid.fit(X, y).best_estimator_
        joblib.dump(model, 'xgb_model.joblib')
        return model, False

# Main function
def main():
    st.title('Loan Default Prediction App')
    df = load_data('Loan_default.csv')
    model, model_loaded = get_model()

    # Data Overview
    with st.expander("🔍 Data Overview", expanded=False):
        overview = st.selectbox(
            "Select what to display:",
            ["Total Missing Values", "Data Head", "Data Types", "Descriptive Statistics"]
        )
        if overview == "Total Missing Values":
            st.dataframe(df.isnull().sum().rename("Missing Count").to_frame())
        elif overview == "Data Head":
            n = st.slider("Rows to view:", 5, 50, 5, 5)
            st.dataframe(df.head(n))
        elif overview == "Data Types":
            st.dataframe(df.dtypes.rename("Dtype").to_frame())
        else:
            st.dataframe(df.describe())

    # EDA Section
    with st.expander("Exploratory Data Analysis (EDA)", expanded=False):
        st.subheader('Class Distribution')
        orig, bal = get_balanced_counts(df)
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots()
            sns.barplot(x=orig.index, y=orig.values, ax=ax1)
            ax1.set_title('Original Data')
            st.pyplot(fig1, use_container_width=True)
        with col2:
            fig2, ax2 = plt.subplots()
            sns.barplot(x=bal.index, y=bal.values, ax=ax2)
            ax2.set_title('After SMOTE')
            st.pyplot(fig2, use_container_width=True)

        st.subheader('Correlation Matrix')
        num_cols = df.select_dtypes(include=np.number).columns.drop('Default')
        corr = df[num_cols].corr()
        fig3, ax3 = plt.subplots()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', ax=ax3)
        st.pyplot(fig3, use_container_width=True)

        st.subheader('Histograms')
        cols_per_row = 3
        for i in range(0, len(num_cols), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(num_cols[i:i+cols_per_row]):
                with cols[j]:
                    fig_hist, ax_hist = plt.subplots()
                    sns.histplot(df[col], kde=True, ax=ax_hist)
                    ax_hist.set_title(f'{col} Distribution')
                    st.pyplot(fig_hist, use_container_width=True)

    # Model Evaluation
    with st.expander("Model Evaluation", expanded=False):
        X = df.drop(columns=['LoanID', 'Default'])
        y = df['Default']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax_cm)
        st.pyplot(fig_cm, use_container_width=True)

        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.legend()
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        st.pyplot(fig_roc, use_container_width=True)

        st.subheader("Feature Importance")
        st.pyplot(plot_feature_importance(model), use_container_width=True)

    # Prediction
    with st.expander("Prediction Result", expanded=False):
        inp_df = user_input_sidebar(df, model_loaded)
        if st.sidebar.button("Predict Default"):
            pred = model.predict(inp_df[ALL_FEATURES])[0]
            prob = model.predict_proba(inp_df[ALL_FEATURES])[0][1]
            result = "Likely to Default" if pred else "Not Likely to Default"
            st.dataframe(pd.DataFrame({
                'Prediction': [result],
                'Probability of Default': [f"{prob:.2%}"]
            }))

if __name__ == '__main__':
    main()