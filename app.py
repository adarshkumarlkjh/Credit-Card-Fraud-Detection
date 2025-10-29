import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Explainability
import shap

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---

@st.cache_data
def get_sample_data():
    """
    Generates a small, synthetic, but representative sample dataset.
    The real dataset is too large to embed.
    """
    np.random.seed(42)
    n_rows = 2000
    n_fraud = 20
    
    # Generate non-fraudulent data
    data = np.random.randn(n_rows, 28)
    df = pd.DataFrame(data, columns=[f'V{i}' for i in range(1, 29)])
    df['Time'] = np.random.randint(1, 172800, n_rows)
    df['Amount'] = np.round(np.abs(np.random.normal(100, 50, n_rows)), 2)
    df['Class'] = 0
    
    # Generate fraudulent data
    fraud_indices = np.random.choice(df.index, n_fraud, replace=False)
    df.loc[fraud_indices, 'V1'] = np.random.normal(-5, 2, n_fraud)
    df.loc[fraud_indices, 'V2'] = np.random.normal(4, 2, n_fraud)
    df.loc[fraud_indices, 'V4'] = np.random.normal(3, 1, n_fraud)
    df.loc[fraud_indices, 'Amount'] = np.round(np.abs(np.random.normal(500, 150, n_fraud)), 2)
    df.loc[fraud_indices, 'Class'] = 1
    
    return df.sample(frac=1).reset_index(drop=True)

@st.cache_data
def load_data(uploaded_file):
    """Loads data from file upload or sample."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    else:
        return get_sample_data()

@st.cache_resource
def preprocess_data(df):
    """Applies preprocessing steps to the data."""
    df_proc = df.copy()
    
    # Assuming 'Time' and 'Amount' are the only non-scaled features
    # 'Class' is the target
    if 'Time' in df_proc.columns and 'Amount' in df_proc.columns:
        scaler = StandardScaler()
        df_proc[['Time', 'Amount']] = scaler.fit_transform(df_proc[['Time', 'Amount']])
        
        X = df_proc.drop('Class', axis=1)
        y = df_proc['Class']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Store scaler
        st.session_state.scaler = scaler
        st.session_state.X_train_cols = X.columns.tolist()
        
        return X_train, X_test, y_train, y_test
    else:
        st.error("Dataset must contain 'Time', 'Amount', and 'Class' columns.")
        return None, None, None, None

@st.cache_resource
def train_model(_model_name, X_train, y_train):
    """Trains a specified machine learning model."""
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(random_state=42, n_jobs=-1),
        "CatBoost": cb.CatBoostClassifier(random_state=42, verbose=0)
    }
    
    model = models.get(_model_name)
    start_time = time.time()
    
    # Handle CatBoost feature names
    if _model_name == "CatBoost" and isinstance(X_train, pd.DataFrame):
        X_train_cat = X_train.copy()
        X_train_cat.columns = [str(c) for c in X_train.columns]
    else:
        X_train_cat = X_train

    model.fit(X_train_cat, y_train)
    end_time = time.time()
    
    train_time = end_time - start_time
    return model, train_time

def evaluate_model(model, X_test, y_test):
    """Generates evaluation metrics for the model."""
    
    # Handle CatBoost feature names
    if isinstance(model, cb.CatBoostClassifier) and isinstance(X_test, pd.DataFrame):
        X_test_cat = X_test.copy()
        X_test_cat.columns = [str(c) for c in X_test.columns]
    else:
        X_test_cat = X_test

    y_pred = model.predict(X_test_cat)
    y_pred_proba = model.predict_proba(X_test_cat)[:, 1]
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_pred_proba),
        "CM": confusion_matrix(y_test, y_pred),
        "y_pred_proba": y_pred_proba
    }
    return metrics

def plot_confusion_matrix(cm):
    """Plots a confusion matrix using Seaborn."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    return fig

def plot_roc_curve(y_test, y_pred_proba):
    """Plots an interactive ROC curve using Plotly."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr,
                        mode='lines',
                        name=f'ROC Curve (AUC = {auc:.4f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                        mode='lines',
                        line=dict(dash='dash'),
                        name='Random Guess'))
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate'
    )
    return fig

def get_model_download_link(model, model_name):
    """Generates a download link for the trained model."""
    output = BytesIO()
    joblib.dump(model, output)
    b64 = base64.b64encode(output.getvalue()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{model_name.lower().replace(" ", "_")}.pkl">Download Trained Model (.pkl)</a>'

def get_report_download_link(df, model_name):
    """Generates a download link for the metrics report."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:text/csv;base64,{b64}" download="{model_name}_evaluation_report.csv">Download Metrics Report (.csv)</a>'

def run_simulation(model, scaler, columns):
    """Simulates real-time transaction classification."""
    placeholder = st.empty()
    st.session_state.simulation_running = True
    
    for i in range(20):
        if not st.session_state.get('simulation_running', False):
            break
            
        # Generate a random transaction
        is_fraud = np.random.rand() < 0.05  # 5% chance of being fraud for simulation
        
        if is_fraud:
            sim_data = {
                'V1': np.random.normal(-5, 2), 'V2': np.random.normal(4, 2),
                'V4': np.random.normal(3, 1), 'Amount': np.round(np.abs(np.random.normal(500, 150)), 2),
                'Class': 1
            }
        else:
            sim_data = {
                'V1': np.random.normal(0, 1), 'V2': np.random.normal(0, 1),
                'V4': np.random.normal(0, 1), 'Amount': np.round(np.abs(np.random.normal(100, 50)), 2),
                'Class': 0
            }
            
        # Fill other columns
        for col in columns:
            if col not in sim_data and col not in ['Time', 'Amount', 'Class']:
                sim_data[col] = np.random.normal(0, 1)
        
        sim_data['Time'] = np.random.randint(1, 172800)
        
        # Create DataFrame and preprocess
        sim_df = pd.DataFrame([sim_data])
        sim_df_features = sim_df[columns]
        sim_df_scaled_features = sim_df_features.copy()
        sim_df_scaled_features[['Time', 'Amount']] = scaler.transform(sim_df_features[['Time', 'Amount']])
        
        # Predict
        if isinstance(model, cb.CatBoostClassifier):
            sim_df_scaled_features.columns = [str(c) for c in sim_df_scaled_features.columns]
            
        pred = model.predict(sim_df_scaled_features)[0]
        prob = model.predict_proba(sim_df_scaled_features)[0]
        
        # Display result
        amount_val = sim_data['Amount']
        if pred == 1:
            placeholder.error(
                f"🚨 FRAUD DETECTED! | Amount: ${amount_val:,.2f} | Confidence: {prob[1]*100:.2f}%",
                icon="🚨"
            )
        else:
            placeholder.success(
                f"✅ Transaction OK | Amount: ${amount_val:,.2f} | Confidence: {prob[0]*100:.2f}%",
                icon="✅"
            )
        
        time.sleep(np.random.uniform(0.5, 2.0))
        
    placeholder.info("Simulation finished.")
    st.session_state.simulation_running = False

# --- Streamlit App UI ---

# Sidebar Navigation
st.sidebar.title("💳 Fraud Detection System")
navigation = st.sidebar.radio(
    "Navigation",
    ["📘 Overview", "📊 EDA & Data Loader", "⚙️ Model Training", "🧠 Explainability", "🔍 Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This app demonstrates a complete machine learning pipeline for "
    "credit card fraud detection, from data loading to prediction and explainability."
)

# --- Page 1: Overview ---
if navigation == "📘 Overview":
    st.title("Credit Card Fraud Detection System")
    st.image(
        "https://images.unsplash.com/photo-1593696890906-19d4b611f486?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80",
        caption="Protecting digital transactions",
        use_column_width=True
    )
    
    st.header("Project Overview")
    st.markdown("""
    Credit card fraud is a significant and growing problem, costing consumers and financial institutions billions of dollars annually.
    Machine learning offers a powerful tool to detect and prevent fraudulent transactions in real-time.
    
    This application allows you to:
    - **Load and explore** transaction data.
    - **Train and compare** multiple state-of-the-art machine learning models.
    - **Evaluate** model performance using comprehensive metrics.
    - **Understand** model decisions using SHAP explainability.
    - **Predict** fraud on new, unseen transactions.
    """)
    
    st.header("Dataset")
    st.markdown("""
    The application can use a sample dataset or a user-uploaded CSV. The sample dataset is synthetically generated
    but mirrors the structure of the famous "Credit Card Fraud Detection" dataset from Kaggle.
    
    **Key features:**
    - `Time`: Seconds elapsed between this transaction and the first transaction in the dataset.
    - `Amount`: Transaction amount.
    - `V1` - `V28`: Anonymized features, likely the result of PCA (Principal Component Analysis).
    - `Class`: The target variable (1 for fraud, 0 for non-fraud).
    """)
    
    st.header("Model Objective")
    st.markdown("""
    The primary goal is to build a classification model that accurately identifies **fraudulent** transactions
    (Class = 1) while minimizing false positives (classifying a normal transaction as fraud).
    
    Given the highly imbalanced nature of the data (fraud is rare), we will focus on metrics like
    **Precision**, **Recall**, **F1-Score**, and **ROC-AUC** over simple accuracy.
    """)

# --- Page 2: EDA & Data Loader ---
elif navigation == "📊 EDA & Data Loader":
    st.header("Exploratory Data Analysis & Data Loader")
    
    # --- 1. Data Loading ---
    st.subheader("1. Load Data")
    data_source = st.radio("Choose data source", ["Use Sample Dataset", "Upload your own CSV"])
    
    uploaded_file = None
    if data_source == "Upload your own CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    df = load_data(uploaded_file)
    
    if df is not None:
        st.session_state.data = df
        st.success("Data loaded successfully!")
        
        # --- 2. Data Preview ---
        st.subheader("2. Data Preview")
        st.dataframe(df.head())
        
        # --- 3. Dataset Summary ---
        st.subheader("3. Dataset Summary")
        col1, col2 = st.columns(2)
        col1.metric("Number of Rows", f"{df.shape[0]:,}")
        col1.metric("Number of Columns", f"{df.shape[1]:,}")
        
        with col2:
            st.write("Missing Values:")
            missing_vals = df.isnull().sum()
            st.dataframe(missing_vals[missing_vals > 0], use_container_width=True)
            if missing_vals.sum() == 0:
                st.write("No missing values found.")
        
        # --- 4. Class Distribution ---
        st.subheader("4. Class Distribution")
        if 'Class' in df.columns:
            class_counts = df['Class'].value_counts()
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Non-Fraudulent (0)", f"{class_counts.get(0, 0):,}")
                st.metric("Fraudulent (1)", f"{class_counts.get(1, 0):,}")
                
            with col2:
                fig_pie = px.pie(
                    class_counts, 
                    values=class_counts.values, 
                    names=class_counts.index.map({0: 'Non-Fraud', 1: 'Fraud'}), 
                    title="Class Distribution",
                    color=class_counts.index.map({0: 'Non-Fraud', 1: 'Fraud'}),
                    color_discrete_map={'Non-Fraud': 'royalblue', 'Fraud': 'darkred'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("Target column 'Class' not found in the dataset.")
            
        # --- 5. Visualization Dashboard (EDA) ---
        st.subheader("5. Exploratory Data Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Distributions", "Box Plots", "Correlation Heatmap"])
        
        with tab1:
            st.write("#### Distribution of Transaction Time & Amount by Class")
            if 'Time' in df.columns:
                fig_time = px.histogram(df, x='Time', color='Class', nbins=100,
                                        title='Transaction Time Distribution',
                                        color_discrete_map={0: 'royalblue', 1: 'darkred'})
                st.plotly_chart(fig_time, use_container_width=True)
            
            if 'Amount' in df.columns:
                fig_amount = px.histogram(df, x='Amount', color='Class', nbins=100,
                                          title='Transaction Amount Distribution (Log Scale)',
                                          log_y=True,
                                          color_discrete_map={0: 'royalblue', 1: 'darkred'})
                st.plotly_chart(fig_amount, use_container_width=True)
                st.caption("Note: Amount distribution uses a log scale on the y-axis due to high variance.")
        
        with tab2:
            st.write("#### Box Plots for Amount and Time by Class")
            if 'Amount' in df.columns:
                fig_box_amount = px.box(df, x='Class', y='Amount', 
                                        title='Transaction Amount by Class',
                                        color='Class',
                                        color_discrete_map={0: 'royalblue', 1: 'darkred'})
                st.plotly_chart(fig_box_amount, use_container_width=True)
            
            if 'Time' in df.columns:
                fig_box_time = px.box(df, x='Class', y='Time', 
                                      title='Transaction Time by Class',
                                      color='Class',
                                      color_discrete_map={0: 'royalblue', 1: 'darkred'})
                st.plotly_chart(fig_box_time, use_container_width=True)

        with tab3:
            st.write("#### Correlation Heatmap")
            st.info("Calculating correlation... This may take a moment for large datasets.")
            
            # Select a subset of features for readability
            v_features = [f'V{i}' for i in range(1, 11)]
            cols_to_corr = v_features + ['Time', 'Amount', 'Class']
            
            if all(col in df.columns for col in cols_to_corr):
                corr_matrix = df[cols_to_corr].corr()
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmin=-1, zmax=1
                ))
                fig_corr.update_layout(title="Correlation Matrix (Subset of Features)")
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("Could not generate heatmap. Ensure V1-V10, Time, Amount, and Class columns exist.")
    else:
        st.info("Please load a dataset or use the sample data to begin analysis.")

# --- Page 3: Model Training ---
elif navigation == "⚙️ Model Training":
    st.header("Model Training & Evaluation")
    
    if 'data' not in st.session_state:
        st.warning("Please load data in the '📊 EDA & Data Loader' section first.")
        st.stop()
        
    # --- 1. Preprocessing ---
    st.subheader("1. Data Preprocessing")
    if st.button("Run Preprocessing (Scale & Split Data)"):
        with st.spinner("Scaling features and splitting data..."):
            X_train, X_test, y_train, y_test = preprocess_data(st.session_state.data)
            if X_train is not None:
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.success("Data preprocessing complete!")
                
                st.json({
                    "Training Set Shape (X)": X_train.shape,
                    "Training Set Shape (y)": y_train.shape,
                    "Testing Set Shape (X)": X_test.shape,
                    "Testing Set Shape (y)": y_test.shape
                })
            else:
                st.error("Preprocessing failed. Check your dataset format.")
    
    if 'X_train' not in st.session_state:
        st.info("Please run preprocessing before training a model.")
        st.stop()
        
    # --- 2. Model Selection & Training ---
    st.subheader("2. Model Selection & Training")
    
    model_option = st.selectbox(
        "Choose a model or AutoML",
        ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost", "🤖 AutoML (Compare All)"]
    )
    
    if st.button(f"Train {model_option} Model"):
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        if model_option == "🤖 AutoML (Compare All)":
            models_to_train = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"]
            leaderboard = []
            
            progress_bar = st.progress(0, text="AutoML in progress...")
            
            for i, model_name in enumerate(models_to_train):
                with st.spinner(f"Training {model_name}..."):
                    model, train_time = train_model(model_name, X_train, y_train)
                    metrics = evaluate_model(model, X_test, y_test)
                    
                    metrics_entry = {
                        "Model": model_name,
                        "Accuracy": metrics["Accuracy"],
                        "Precision": metrics["Precision"],
                        "Recall": metrics["Recall"],
                        "F1-Score": metrics["F1-Score"],
                        "ROC-AUC": metrics["ROC-AUC"],
                        "Train Time (s)": train_time
                    }
                    leaderboard.append(metrics_entry)
                
                progress_bar.progress((i + 1) / len(models_to_train), text=f"Trained {model_name}")
            
            progress_bar.empty()
            st.success("AutoML training complete!")
            
            # Display Leaderboard
            leaderboard_df = pd.DataFrame(leaderboard).sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
            st.session_state.leaderboard = leaderboard_df
            
            st.subheader("AutoML Leaderboard")
            st.dataframe(leaderboard_df, use_container_width=True)
            
            # Save the best model
            best_model_name = leaderboard_df.iloc[0]["Model"]
            st.info(f"🏆 Best model found: **{best_model_name}**. Saving this model.")
            best_model, _ = train_model(best_model_name, X_train, y_train)
            st.session_state.model = best_model
            st.session_state.model_name = best_model_name
            st.session_state.metrics = evaluate_model(best_model, X_test, y_test)
            st.session_state.metrics_df = pd.DataFrame([leaderboard_df.iloc[0]])

        else:
            # Train a single model
            with st.spinner(f"Training {model_option}... This may take a minute."):
                model, train_time = train_model(model_option, X_train, y_train)
                st.session_state.model = model
                st.session_state.model_name = model_option
                
                metrics = evaluate_model(model, X_test, y_test)
                st.session_state.metrics = metrics
                
                metrics_entry = {
                    "Model": model_option,
                    "Accuracy": metrics["Accuracy"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "F1-Score": metrics["F1-Score"],
                    "ROC-AUC": metrics["ROC-AUC"],
                    "Train Time (s)": train_time
                }
                st.session_state.metrics_df = pd.DataFrame([metrics_entry])

            st.success(f"{model_option} trained successfully in {train_time:.2f} seconds!")

    # --- 3. Evaluation ---
    if 'metrics' in st.session_state:
        st.subheader("3. Model Evaluation")
        st.info(f"Displaying metrics for the selected model: **{st.session_state.model_name}**")
        
        metrics = st.session_state.metrics
        
        # Display key metrics
        st.write("#### Key Performance Indicators (KPIs)")
        cols = st.columns(5)
        cols[0].metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        cols[1].metric("Precision", f"{metrics['Precision']:.4f}")
        cols[2].metric("Recall", f"{metrics['Recall']:.4f}")
        cols[3].metric("F1-Score", f"{metrics['F1-Score']:.4f}")
        cols[4].metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}")
        
        # Display plots
        st.write("#### Performance Plots")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Confusion Matrix**")
            fig_cm = plot_confusion_matrix(metrics['CM'])
            st.pyplot(fig_cm)
            
        with col2:
            st.write("**ROC Curve**")
            fig_roc = plot_roc_curve(st.session_state.y_test, metrics['y_pred_proba'])
            st.plotly_chart(fig_roc, use_container_width=True)
            
    # --- 4. Download Model & Report ---
    if 'model' in st.session_state and 'metrics_df' in st.session_state:
        st.subheader("4. Download Artifacts")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                get_model_download_link(st.session_state.model, st.session_state.model_name),
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                get_report_download_link(st.session_state.metrics_df, st.session_state.model_name),
                unsafe_allow_html=True
            )

# --- Page 4: Explainability ---
elif navigation == "🧠 Explainability":
    st.header("Model Explainability (SHAP)")
    
    if 'model' not in st.session_state:
        st.warning("Please train a model in the '⚙️ Model Training' section first.")
        st.stop()
        
    model = st.session_state.model
    model_name = st.session_state.model_name
    X_test = st.session_state.X_test
    
    st.info(f"Generating explanations for the **{model_name}** model using SHAP.")
    
    # SHAP calculation
    if st.button("Calculate SHAP Values"):
        with st.spinner("Calculating SHAP values... This can be slow, especially for non-tree models."):
            X_test_sample = shap.sample(X_test, 100) # Use a sample for speed
            
            if model_name in ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]:
                # Handle CatBoost feature names
                if model_name == "CatBoost":
                    X_test_sample.columns = [str(c) for c in X_test_sample.columns]
                
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_sample)
                
                # For binary classification, shap_values can be a list of [class_0, class_1]
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] # Use values for the positive class (Fraud)
                    
            else: # Logistic Regression
                explainer = shap.LinearExplainer(model, X_test) # Use full X_test for masker
                shap_values = explainer.shap_values(X_test_sample)
            
            st.session_state.shap_values = shap_values
            st.session_state.X_test_sample = X_test_sample
            st.success("SHAP values calculated successfully!")

    # Display SHAP plots
    if 'shap_values' in st.session_state:
        st.subheader("SHAP Summary Plot")
        st.info("This plot shows the most important features. Each point is a transaction. "
                "Red points = high feature value, Blue points = low feature value. "
                "Positive SHAP value = increases prediction of fraud.")
        
        fig_summary, ax_summary = plt.subplots()
        shap.summary_plot(
            st.session_state.shap_values,
            st.session_state.X_test_sample,
            show=False,
            plot_type='dot'
        )
        st.pyplot(fig_summary)
        
        st.subheader("SHAP Feature Importance (Bar Plot)")
        st.info("This plot shows the average impact of each feature on the model's output magnitude (i.e., feature importance).")
        
        fig_bar, ax_bar = plt.subplots()
        shap.summary_plot(
            st.session_state.shap_values,
            st.session_state.X_test_sample,
            show=False,
            plot_type='bar'
        )
        st.pyplot(fig_bar)
        
        st.subheader("SHAP Dependence Plot")
        st.info("Select a feature to see its effect on the prediction and its interaction with another feature.")
        
        all_features = st.session_state.X_test_sample.columns.tolist()
        feature_to_plot = st.selectbox("Select feature to plot", all_features, index=all_features.index('V4') if 'V4' in all_features else 0)
        
        fig_dep, ax_dep = plt.subplots()
        shap.dependence_plot(
            feature_to_plot,
            st.session_state.shap_values,
            st.session_state.X_test_sample,
            show=False,
            interaction_index="auto" # Automatically finds the most interactive feature
        )
        st.pyplot(fig_dep)
        
# --- Page 5: Prediction ---
elif navigation == "🔍 Prediction":
    st.header("Make a Real-time Prediction")
    
    if 'model' not in st.session_state or 'scaler' not in st.session_state:
        st.warning("Please train a model in the '⚙️ Model Training' section first.")
        st.stop()
        
    model = st.session_state.model
    scaler = st.session_state.scaler
    columns = st.session_state.X_train_cols

    # --- 1. Manual Input Form ---
    st.subheader("1. Manual Transaction Input")
    st.info("Enter transaction details below. V1-V28 are anonymized features.")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            time_input = st.number_input("Time (seconds since first transaction)", min_value=0, value=172000)
        with col2:
            amount_input = st.number_input("Amount ($)", min_value=0.0, value=100.0, format="%.2f")
        
        v_features = {}
        with st.expander("Enter V1-V28 Features"):
            cols_v = st.columns(4)
            for i in range(1, 29):
                # Use default values based on fraud/non-fraud averages if known
                # Here, we just default to 0.0
                v_features[f'V{i}'] = cols_v[(i-1)%4].number_input(f"V{i}", value=0.0, key=f"v{i}", format="%.6f")
        
        submit_button = st.form_submit_button("Predict Fraud")
        
    if submit_button:
        # Create input DataFrame
        input_data = v_features
        input_data['Time'] = time_input
        input_data['Amount'] = amount_input
        
        try:
            input_df = pd.DataFrame([input_data])
            input_df = input_df[columns] # Ensure correct column order
            
            # Scale data
            input_df_scaled = input_df.copy()
            input_df_scaled[['Time', 'Amount']] = scaler.transform(input_df[['Time', 'Amount']])
            
            # Handle CatBoost feature names
            if isinstance(model, cb.CatBoostClassifier):
                input_df_scaled.columns = [str(c) for c in input_df_scaled.columns]
            
            # Predict
            prediction = model.predict(input_df_scaled)[0]
            prediction_proba = model.predict_proba(input_df_scaled)[0]
            
            # Display result
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"🔴 Prediction: FRAUDULENT Transaction", icon="🚨")
                st.metric("Fraud Confidence", f"{prediction_proba[1]*100:.2f}%")
                st.balloons()
            else:
                st.success(f"🟢 Prediction: Non-Fraudulent Transaction", icon="✅")
                st.metric("Non-Fraud Confidence", f"{prediction_proba[0]*100:.2f}%")
                
            with st.expander("Show Raw Prediction Data"):
                st.write("Scaled Input Data:")
                st.dataframe(input_df_scaled)
                st.write("Prediction Probabilities:", {"Non-Fraud (0)": prediction_proba[0], "Fraud (1)": prediction_proba[1]})
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("Ensure all V1-V28 features are entered, or that the model was trained correctly.")

    # --- 2. Real-time Simulation ---
    st.subheader("2. Real-time Detection Simulation")
    st.info("Click 'Start' to simulate a stream of incoming transactions and classify them live.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Simulation", key="start_sim", disabled=st.session_state.get('simulation_running', False)):
            run_simulation(model, scaler, columns)
    with col2:
        if st.button("Stop Simulation", key="stop_sim", disabled=not st.session_state.get('simulation_running', False)):
            st.session_state.simulation_running = False
            st.info("Simulation stopping...")
