import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Iris Intelligence Suite",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD PICKLE FILES & DATASET
# =========================================================
@st.cache_resource
def load_assets():
    with open("iris_kmeans_model.pkl", "rb") as file:
        data = pickle.load(file)
    return data["model"], data["scaler"], data["features"]

try:
    model, scaler, feature_columns = load_assets()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# =========================================================
# STYLES & COLOR PALETTE (EMERALD GLASSMORPHISM)
# =========================================================
BG_IMAGE_URL = "https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?q=80&w=1920&auto=format&fit=crop"

st.markdown(
    f"""
    <style>
    /* Background setup */
    .stApp {{
        background-image: url("{BG_IMAGE_URL}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* Global Typography */
    h1, h2, h3, h4, h5, h6, p, label, span, .stMarkdown {{
        color: #F8FAFC !important;
        font-weight: 600 !important;
    }}

    /* Center Align Tab Bar Container */
    .stTabs [data-baseweb="tab-list"] {{
        justify-content: center !important;
        gap: 20px !important;
        margin-bottom: 16px !important;
    }}

    /* High Specificity Tab Button Styling */
    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] {{
        padding: 14px 36px !important;
        border-radius: 14px !important;
        background-color: rgba(15, 23, 42, 0.8) !important;
        box-shadow: 0px 4px 14px rgba(0, 0, 0, 0.4) !important;
        border: 1px solid rgba(16, 185, 129, 0.4) !important;
        height: auto !important;
    }}

    /* Enforce Large Font Size across all sub-elements in Tabs */
    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] * {{
        font-size: 24px !important;
        font-weight: 800 !important;
        color: #34D399 !important;
        line-height: 1.2 !important;
    }}

    /* Expanded & Colored Input Form Container */
    div[data-testid="stForm"] {{
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(6, 78, 59, 0.85) 100%) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        padding: 35px !important;
        border-radius: 20px !important;
        box-shadow: 0px 12px 35px rgba(0, 0, 0, 0.6) !important;
        border: 2px solid #10B981 !important;
        min-height: 480px;
    }}

    /* Tab Content Panel Styling with Backdrop Blur */
    .stTabs [data-baseweb="tab-panel"] {{
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        padding: 32px !important;
        border-radius: 20px !important;
        box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.4) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
    }}

    /* Hero Header for Top Title */
    .hero-container {{
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.92) 0%, rgba(4, 47, 46, 0.92) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 32px 28px;
        border-radius: 18px;
        color: #FFFFFF !important;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0px 8px 25px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(52, 211, 153, 0.4);
    }}
    .hero-container h1, .hero-container h3, .hero-container div {{
        color: #FFFFFF !important;
    }}
    
    /* Dashboard Title */
    .hero-title {{
        font-size: 50px !important;
        font-weight: 900 !important;
        margin: 0;
        color: #34D399 !important;
        letter-spacing: -0.5px;
    }}
    
    .hero-subtitle {{
        font-size: 18px;
        margin-top: 8px;
        color: #A7F3D0 !important;
    }}

    /* Prediction Output Banner */
    .result-card {{
        background: linear-gradient(135deg, #047857 0%, #065F46 100%);
        padding: 28px;
        border-radius: 18px;
        color: #FFFFFF !important;
        text-align: center;
        margin: 24px 0;
        box-shadow: 0px 10px 28px rgba(4, 120, 87, 0.5);
        border: 1px solid #34D399;
    }}
    .result-card div, .result-card h1, .result-card p {{
        color: #FFFFFF !important;
    }}
    .result-value {{
        font-size: 42px;
        font-weight: 900;
        margin: 6px 0;
        color: #6EE7B7 !important;
    }}

    /* About Us Cards with Blur */
    .card {{
        background: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        padding: 28px;
        border-radius: 16px;
        box-shadow: 0px 6px 18px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(52, 211, 153, 0.4);
    }}

    /* Input & Select Box Accent Colors */
    div[data-baseweb="select"] > div, input {{
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 2px solid #10B981 !important;
        border-radius: 8px !important;
    }}

    /* Section Heading Panels */
    .section-header {{
        font-size: 24px !important;
        font-weight: 900 !important;
        color: #34D399 !important;
        margin-bottom: 20px !important;
        border-bottom: 3px solid #10B981 !important;
        padding-bottom: 10px !important;
        letter-spacing: 0.5px;
    }}

    /* About Us Footer Container with Blur */
    .footer-container {{
        text-align: center;
        padding: 25px;
        background: rgba(2, 44, 34, 0.9);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        box-shadow: 0px 8px 25px rgba(0, 0, 0, 0.5);
        margin-top: 24px;
        border: 1px solid #10B981;
    }}
    .footer-container h3 {{
        color: #F8FAFC !important;
        font-size: 22px;
        margin-bottom: 8px;
    }}
    .footer-container p {{
        color: #CBD5E1 !important;
        font-size: 15px;
        margin: 4px 0;
    }}
    .footer-container span {{
        color: #34D399 !important;
        font-weight: bold !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# DASHBOARD HEADER
# =========================================================
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🌿 Iris Intelligent Cluster Suite</div>
        <div class="hero-subtitle">High-Performance Botanical Pattern Recognition & Machine Learning Architecture</div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# MODEL CHECK
# =========================================================
if not model_loaded:
    st.error("⚠️ Pickle file 'iris_kmeans_model.pkl' not found! Please check model training output files.")
    st.stop()

# =========================================================
# CONTROL PANEL & INPUT FORM
# =========================================================
tab_predict, tab_about = st.tabs(["📊 Prediction Form", "ℹ️ About Us"])

with tab_predict:
    with st.form(key="prediction_form"):
        col_sepal, col_petal = st.columns(2, gap="large")

        # SEPAL INPUTS
        with col_sepal:
            st.markdown('<div class="section-header">🌿 Sepal Morphometrics</div>', unsafe_allow_html=True)
            sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
            sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)

        # PETAL INPUTS
        with col_petal:
            st.markdown('<div class="section-header">🌷 Petal Morphometrics</div>', unsafe_allow_html=True)
            petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
            petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

        st.write("")
        predict_button = st.form_submit_button("🔍 Execute Cluster Inference", use_container_width=True, type="primary")

    # =========================================================
    # INFERENCE & DASHBOARD VISUALS (PREDICTION TAB ONLY)
    # =========================================================
    if predict_button:
        input_data = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=feature_columns
        )

        input_scaled = scaler.transform(input_data)
        cluster = model.predict(input_scaled)[0]
        distances = model.transform(input_scaled)[0]
        closest_distance = distances[cluster]

        st.markdown(
            f"""
            <div class="result-card">
                <div style="font-size: 16px; text-transform: uppercase; letter-spacing: 1.2px;">Unsupervised Cluster Assignment</div>
                <div class="result-value">Cluster {cluster}</div>
                <div style="font-size: 15px; margin-top: 5px;">Euclidean Distance to Centroid: <b>{closest_distance:.4f}</b></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sepal Length", f"{sepal_length:.1f} cm")
        m2.metric("Sepal Width", f"{sepal_width:.1f} cm")
        m3.metric("Petal Length", f"{petal_length:.1f} cm")
        m4.metric("Petal Width", f"{petal_width:.1f} cm")

        st.markdown("### 📈 Cluster Distribution & Centroid Analysis")
        graph_col1, graph_col2 = st.columns(2)

        with graph_col1:
            centers_scaled = model.cluster_centers_
            centers_original = scaler.inverse_transform(centers_scaled)
            centers_df = pd.DataFrame(centers_original, columns=feature_columns)
            centers_df["Cluster"] = [f"Cluster {i}" for i in range(model.n_clusters)]
            
            fig_radar = go.Figure()
            for idx, row in centers_df.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row[feature_columns[0]], row[feature_columns[1]], row[feature_columns[2]], row[feature_columns[3]]],
                    theta=feature_columns,
                    fill='toself',
                    name=row["Cluster"]
                ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 8], color="white"),
                    angularaxis=dict(color="white"),
                    bgcolor="#0F172A"
                ),
                title=dict(text="Cluster Centroid Profiles", font=dict(color="white")),
                height=360,
                paper_bgcolor='#0F172A',
                font={'color': "#F8FAFC"},
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with graph_col2:
            iris_data = load_iris()
            df_full = pd.DataFrame(iris_data.data, columns=feature_columns)
            df_full["Cluster"] = model.predict(scaler.transform(df_full))
            df_full["Cluster_Name"] = df_full["Cluster"].astype(str)

            fig_scatter = px.scatter(
                df_full,
                x=feature_columns[2],
                y=feature_columns[3],
                color="Cluster_Name",
                title="Petal Dimensions vs K-Means Clusters",
                color_discrete_sequence=["#34D399", "#10B981", "#059669"]
            )
            # Add user point
            fig_scatter.add_trace(go.Scatter(
                x=[petal_length],
                y=[petal_width],
                mode="markers",
                marker=dict(color="#F59E0B", size=16, symbol="x"),
                name="Your Input"
            ))
            fig_scatter.update_layout(
                height=360, 
                paper_bgcolor='#0F172A', 
                plot_bgcolor='#1E293B', 
                font={'color': "#F8FAFC"}, 
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

# =========================================================
# ABOUT US TAB
# =========================================================
with tab_about:
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.markdown("""
## 📖 About the Project

**Iris Intelligent Cluster Suite** is an advanced Unsupervised Machine Learning application designed to group Iris specimens using K-Means clustering based on physical flower measurements.

Botanical classification requires careful observation of sepal and petal features. By applying dimensionality scaling and centroid distance computation, this application highlights natural morphological groupings without relying on pre-existing class labels.

---

## 🚀 Workflow

Flower Measurement Inputs (Sepal & Petal)  
⬇  
Standard Feature Scaling (`StandardScaler`)  
⬇  
Trained K-Means Model Inference (`K = 3`)  
⬇  
Cluster Centroid Distance Calculation  
⬇  
Visualizing Proximity and Feature Profiles  

---

## ✨ Features

✅ Unsupervised K-Means Clustering  
✅ Interactive Morphometric Inputs  
✅ Multi-Feature Radar Centroid Profiling  
✅ High-Contrast Dark Glassmorphic Dashboard UI  
✅ Real-time Distance Metric Evaluation  
""")

    with right:
        st.markdown("""
<div class="card">

## 📌 Project Details

🌿 **Domain**  
Botanical Data Science

🤖 **Model**  
K-Means Clustering

📊 **Dataset**  
Classic Iris Dataset

👥 **Clusters**  
3 Morphological Groups

📈 **Features**  
4 Numerical Measurements

💻 **Framework**  
Streamlit

🐍 **Language**  
Python

</div>
""", unsafe_allow_html=True)

    st.write("---")

    st.subheader("🛠 Technology Stack")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info("""
### 🐍 Python & Data
✔ Python  
✔ NumPy  
✔ Pandas  
✔ Pickle  
""")

    with c2:
        st.info("""
### 🤖 Machine Learning
✔ K-Means Clustering  
✔ Scikit-Learn  
✔ Standard Scaler  
✔ PCA Projection  
""")

    with c3:
        st.info("""
### 🌐 Dashboard & Visuals
✔ Streamlit  
✔ Plotly Charts  
✔ Dark Glassmorphism CSS  
""")

    st.write("---")

    st.subheader("🎯 Clustering Benefits")

    a, b, c = st.columns(3)

    with a:
        st.metric("Unsupervised Pattern Recognition", "100%")

    with b:
        st.metric("Dimensionality Optimization", "4 Features")

    with c:
        st.metric("Centroid Precision", "High Stability")

    st.write("---")

    st.subheader("📈 Why Unsupervised Learning in Botany?")

    st.info("""
Unsupervised learning uncovers hidden structural groupings in biological data without human bias.

Key Advantages:
• Objective cluster creation based purely on numeric variance.
• Efficient classification of mixed populations.
• Automated feature importance tracking across dimensions.
• Scalable pipeline for high-throughput floral analysis.
""")

    st.write("---")

    st.subheader("👨‍💻 Developer")

    st.markdown("""
<div class="card" style="border: 2px solid #10B981;">

<h2 style="color:#34D399;">Aniket Andhale</h2>

<hr style="border: 1px solid rgba(16, 185, 129, 0.3);">

<b>Project :</b> Iris Intelligent Cluster Suite<br><br>
<b>Role :</b> Machine Learning Developer<br><br>
<b>Skills :</b>
<ul>
<li>Python</li>
<li>Machine Learning</li>
<li>Scikit-Learn</li>
<li>Pandas</li>
<li>NumPy</li>
<li>Streamlit</li>
</ul>

</div>
""", unsafe_allow_html=True)

    st.write("")

    st.subheader("📬 Contact")

    st.success("""
📧 Email: aniketandhale30@gmail.com  
💼 LinkedIn: https://www.linkedin.com/in/aniket-andhale-b072b8380  
🐙 GitHub: https://github.com/aniketandhale52-ux  
""")

    st.write("---")

    st.markdown("""
<div class="footer-container">
    <h3>🌿 Iris Intelligent Cluster Suite</h3>
    <p>Built with <span>Aniket Andhale</span> using Python, Streamlit & Machine Learning</p>
    <p>© 2026 Aniket Andhale | All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)