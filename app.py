import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
import base64
from risk_engine import process_machine_data

# Set page configuration for professional dark theme look
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def add_bg_from_local(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(14, 17, 23, 0.8), rgba(14, 17, 23, 0.8)), url(data:image/png;base64,{encoded_string.decode()});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
        )

add_bg_from_local('assets/machine_bg.png')

# Custom CSS for Industrial Styling
st.markdown("""
<style>
    body { color: #fafafa; background-color: transparent !important; }
    [data-testid="stAppViewContainer"] { background-color: transparent !important; }
    [data-testid="stHeader"] { background-color: transparent !important; }
    .kpi-card {
        background-color: rgba(30, 30, 30, 0.85);
        border-left: 5px solid #4CAF50;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.8);
        text-align: left;
        margin-bottom: 20px;
    }
    .kpi-card-high { border-left-color: #ff4b4b; }
    .kpi-card-med { border-left-color: #ffa500; }
    .kpi-card-low { border-left-color: #00fa9a; }
    .kpi-card-warn { border-left-color: #ffaaaa; }
    .kpi-title { font-size: 1rem; color: #a0a0a0; font-weight: 600; text-transform: uppercase; }
    .kpi-value { font-size: 2.2rem; font-weight: bold; margin-top: 5px; }
    .high-risk { color: #ff4b4b !important; }
    .medium-risk { color: #ffa500 !important; }
    .low-risk { color: #00fa9a !important; }
    
    /* Table headers */
    th { background-color: #2b2b2b !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Loading and Processing
# ---------------------------------------------------------
@st.cache_data
def load_data():
    """Load and process the Excel file securely."""
    file_path = "data/machine_data.xlsx"
    if not os.path.exists(file_path):
        return None
    
    try:
        df = pd.read_excel(file_path)
        # Apply risk engine logic
        df = process_machine_data(df)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def load_ml_model(df):
    """Load the ML model if it exists and predict breakdown probability safely."""
    model_path = "models/maintenance_model.pkl"
    if not os.path.exists(model_path) or df.empty:
        return df, False
    
    try:
        model = joblib.load(model_path)
        
        features = [
            'Progressive Work Hours',
            'Progressive Breakdown Hrs',
            'Progressive Maintenance Hrs',
            'Actual Progressive % Availability',
            'Actual Progressive % Utilization',
            'Total Running Hrs.',
            'Life Completed (in Yrs)'
        ]
        
        X = df.copy()
        for col in features:
            if col not in X.columns:
                X[col] = 0
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            
        X = X[features]
        
        if len(model.classes_) >= 2:
            probs = model.predict_proba(X)[:, 1]
        else:
            preds = model.predict(X)
            probs = [1.0 if p == 1 else 0.0 for p in preds]
            
        df['Breakdown Probability %'] = np.round(np.array(probs) * 100, 2)
        return df, True
        
    except Exception as e:
        st.sidebar.warning(f"ML Model error: {e}. Running with Risk Engine only.")
        return df, False

# ---------------------------------------------------------
# Main Application
# ---------------------------------------------------------
def main():
    st.title("⚙️ Industrial Monitoring System")
    st.markdown("Advanced Predictive Maintenance & Machine Health Dashboard")
    
    df = load_data()
        
    if df is None:
        st.error("Dataset not found! Please place `machine_data.xlsx` inside the `data/` folder.")
        return
        
    if df.empty:
        st.warning("The dataset is empty or corrupted.")
        return
        
    df, is_model_loaded = load_ml_model(df)
    
    # ---------------------------------------------------------
    # Sidebar Filters
    # ---------------------------------------------------------
    st.sidebar.header("🔍 Filter Options")
    
    if 'Area Description' in df.columns:
        areas = ["All"] + sorted([str(x) for x in df['Area Description'].dropna().unique()])
        selected_area = st.sidebar.selectbox("Select Area", areas)
        if selected_area != "All":
            df = df[df['Area Description'].astype(str) == selected_area]
            
    if 'Type of Equipment' in df.columns:
        names = ["All"] + sorted([str(x) for x in df['Type of Equipment'].dropna().unique()])
        selected_name = st.sidebar.selectbox("Select Machine Name", names)
        if selected_name != "All":
            df = df[df['Type of Equipment'].astype(str) == selected_name]
            
    if 'CIL Number' in df.columns:
        numbers = ["All"] + sorted([str(x) for x in df['CIL Number'].dropna().unique()])
        selected_number = st.sidebar.selectbox("Select Machine Number", numbers)
        if selected_number != "All":
            df = df[df['CIL Number'].astype(str) == selected_number]
            
    if df.empty:
        st.warning("No machines found matching the selected filters.")
        return

    # ---------------------------------------------------------
    # KPIs Section
    # ---------------------------------------------------------
    st.markdown("### 📊 System Overview")
    
    total_machines = len(df)
    high_risk_count = len(df[df['Priority'] == 'Priority 1'])
    medium_risk_count = len(df[df['Priority'] == 'Priority 2'])
    low_risk_count = len(df[df['Priority'] == 'Priority 3'])
    
    breakdown_count = 0
    if 'Working Status' in df.columns:
        breakdown_count = len(df[df['Working Status'].astype(str).str.upper().str.strip() == 'BREAKDOWN'])

    avg_risk = round(df['Risk Score'].mean(), 1) if 'Risk Score' in df.columns else 0
    immediate_inspections = high_risk_count

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Total Machines</div><div class="kpi-value">{total_machines}</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="kpi-card kpi-card-high"><div class="kpi-title">High Risk (P1)</div><div class="kpi-value high-risk">{high_risk_count}</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="kpi-card kpi-card-med"><div class="kpi-title">Medium Risk (P2)</div><div class="kpi-value medium-risk">{medium_risk_count}</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="kpi-card kpi-card-low"><div class="kpi-title">Low Risk (P3)</div><div class="kpi-value low-risk">{low_risk_count}</div></div>""", unsafe_allow_html=True)

    col5, col6, col7 = st.columns(3)
    with col5:
        st.markdown(f"""<div class="kpi-card kpi-card-warn"><div class="kpi-title">Current Breakdowns</div><div class="kpi-value" style="color:#ffaaaa;">{breakdown_count}</div></div>""", unsafe_allow_html=True)
    with col6:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Average Risk Score</div><div class="kpi-value">{avg_risk}</div></div>""", unsafe_allow_html=True)
    with col7:
        st.markdown(f"""<div class="kpi-card kpi-card-high"><div class="kpi-title">Immediate Inspections</div><div class="kpi-value high-risk">{immediate_inspections}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ---------------------------------------------------------
    # Visualizations
    # ---------------------------------------------------------
    st.markdown("### 📈 Advanced Analytics")
    v_col1, v_col2, v_col3 = st.columns(3)
    
    color_map = {'HIGH':'#ff4b4b', 'MEDIUM':'#ffa500', 'LOW':'#00fa9a'}
    
    with v_col1:
        risk_counts = df['Risk Category'].value_counts().reset_index()
        risk_counts.columns = ['Risk Category', 'Count']
        fig1 = px.pie(risk_counts, values='Count', names='Risk Category', title="Risk Distribution", color='Risk Category', color_discrete_map=color_map, hole=0.4)
        fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig1, use_container_width=True)

    with v_col2:
        if 'Working Status' in df.columns:
            status_counts = df['Working Status'].astype(str).value_counts().reset_index()
            status_counts.columns = ['Working Status', 'Count']
            fig2 = px.pie(status_counts, values='Count', names='Working Status', title="Working Status", hole=0.4)
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig2, use_container_width=True)
            
    with v_col3:
        top_scores = df.nlargest(20, 'Risk Score')
        if 'Equipment' not in top_scores.columns:
            top_scores['Equipment'] = top_scores.index
        fig3 = px.bar(top_scores, x='Equipment', y='Risk Score', color='Risk Category', color_discrete_map=color_map, title="Top 20 Risk Scores")
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig3, use_container_width=True)

    v_col4, v_col5, v_col6 = st.columns(3)
    with v_col4:
        if 'Actual Progressive % Availability' in df.columns:
            fig4 = px.histogram(df, x='Actual Progressive % Availability', nbins=20, title="Availability Distribution", color_discrete_sequence=['#4CAF50'])
            fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig4, use_container_width=True)

    with v_col5:
        if 'Actual Progressive % Utilization' in df.columns:
            fig5 = px.histogram(df, x='Actual Progressive % Utilization', nbins=20, title="Utilization Distribution", color_discrete_sequence=['#2196F3'])
            fig5.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig5, use_container_width=True)

    with v_col6:
        if 'Progressive Breakdown Hrs' in df.columns:
            fig6 = px.histogram(df, x='Progressive Breakdown Hrs', nbins=20, title="Breakdown Hours Distribution", color_discrete_sequence=['#ff4b4b'])
            fig6.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig6, use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------
    # Data Tables
    # ---------------------------------------------------------
    st.markdown("### 📋 Machine Intelligence Tables")
    
    base_cols = ['Risk Score', 'Risk Category', 'Priority', 'Maintenance Recommendation', 'Next Action']
    if 'Equipment' in df.columns:
        base_cols.insert(0, 'Equipment')
    if 'Working Status' in df.columns:
        base_cols.insert(1, 'Working Status')
        
    safe_cols = [c for c in base_cols if c in df.columns]

    tab1, tab2, tab3, tab4 = st.tabs(["🚨 Top 20 Most Critical Machines", "✅ Top 20 Healthiest Machines", "🤖 Predicted Breakdown Risk", "🛠️ Full Maintenance Directory"])
    
    def color_priority(val):
        color = '#00fa9a' if val == 'Priority 3' else '#ffa500' if val == 'Priority 2' else '#ff4b4b' if val == 'Priority 1' else ''
        return f'color: {color}'

    with tab1:
        critical = df.sort_values(by='Risk Score', ascending=False).head(20)[safe_cols]
        st.dataframe(critical.style.map(color_priority, subset=['Priority'] if 'Priority' in critical.columns else []), use_container_width=True)

    with tab2:
        healthy = df.sort_values(by='Risk Score', ascending=True).head(20)[safe_cols]
        st.dataframe(healthy.style.map(color_priority, subset=['Priority'] if 'Priority' in healthy.columns else []), use_container_width=True)

    with tab3:
        if is_model_loaded and 'Breakdown Probability %' in df.columns:
            ml_cols = safe_cols.copy()
            if 'Breakdown Probability %' not in ml_cols:
                ml_cols.insert(2, 'Breakdown Probability %')
            ml_df = df.sort_values(by='Breakdown Probability %', ascending=False)[ml_cols]
            
            # Highlight probabilities > 50%
            def highlight_prob(val):
                return 'color: #ff4b4b; font-weight: bold' if isinstance(val, (int, float)) and val > 50 else ''
                
            st.dataframe(ml_df.style.map(highlight_prob, subset=['Breakdown Probability %']), use_container_width=True)
        else:
            st.warning("Machine Learning model not loaded. Train the model using `python train_model.py` to view Breakdown Probabilities.")

    with tab4:
        st.dataframe(df[safe_cols], use_container_width=True)

if __name__ == "__main__":
    main()
