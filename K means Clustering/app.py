# ============================================================
# K-MEANS CLUSTERING - PROFESSIONAL INTERACTIVE STREAMLIT APP
# Project: Facebook Live Posts Clustering
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="K-Means Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 30px;
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }

    .hero h1 {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero p {
        font-size: 17px;
        opacity: 0.92;
    }

    .metric-card {
        padding: 22px;
        border-radius: 16px;
        background: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.07);
        text-align: center;
        border: 1px solid #edf0f5;
    }

    .metric-title {
        color: #6b7280;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: #111827;
        font-size: 30px;
        font-weight: 800;
        margin-top: 5px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 750;
        color: #111827;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .info-box {
        padding: 18px;
        border-radius: 14px;
        background: #eef4ff;
        border-left: 5px solid #667eea;
        margin: 15px 0;
    }

    .prediction-box {
        padding: 30px;
        border-radius: 18px;
        background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
        border: 1px solid #bbf7d0;
        text-align: center;
        margin-top: 20px;
    }

    .prediction-number {
        font-size: 52px;
        font-weight: 900;
        color: #15803d;
    }

    .prediction-label {
        font-size: 18px;
        font-weight: 700;
        color: #166534;
    }

    .footer {
        text-align: center;
        color: #6b7280;
        margin-top: 40px;
        padding: 20px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------
MODEL_PATH = Path("kmeans_model.pkl")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, "Model file 'kmeans_model.pkl' not found."

    try:
        with open(MODEL_PATH, "rb") as file:
            package = pickle.load(file)

        if isinstance(package, dict):
            model = package.get("model")
            scaler = package.get("scaler")
            label_encoder = package.get("label_encoder")
            features = package.get("features")

            if model is None:
                return None, "K-Means model is missing inside pickle."

            return {
                "model": model,
                "scaler": scaler,
                "label_encoder": label_encoder,
                "features": features
            }, None

        return {
            "model": package,
            "scaler": None,
            "label_encoder": None,
            "features": None
        }, None

    except Exception as e:
        return None, str(e)


model_data, model_error = load_model()


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 K-Means Analytics")
    st.markdown("---")
    st.markdown("### 🧠 Model Status")

    if model_data:
        st.success("Model Loaded Successfully")
    else:
        st.error("Model Not Loaded")

    st.markdown("---")
    st.markdown("### 📌 Project Metadata")
    st.write("**Algorithm:** K-Means Clustering")
    st.write("**Dataset:** Facebook Live Posts")
    st.write("**Type:** Unsupervised Learning")

    st.markdown("---")
    st.markdown("### 🚀 Dashboard Capabilities")
    st.write("• Single Sample Prediction")
    st.write("• Batch CSV Cluster Tagging")
    st.write("• Interactive 2D/3D Scatter Plots")
    st.write("• Cluster Profiles & Centroid Analysis")


# ------------------------------------------------------------
# HERO SECTION
# ------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>📊 K-Means Clustering Analytics</h1>
    <p>
        An interactive machine learning dashboard for exploring
        Facebook Live post clusters using K-Means clustering.
    </p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# MODEL ERROR HANDLING
# ------------------------------------------------------------
if model_error:
    st.error(f"⚠️ {model_error}")
    st.info("""
    Please keep `kmeans_model.pkl` in the same folder as `app.py`.

    Directory Structure:
    Project/
    ├── app.py
    └── kmeans_model.pkl
    """)
    st.stop()


# ------------------------------------------------------------
# MODEL PARAMETERS
# ------------------------------------------------------------
model = model_data["model"]
scaler = model_data["scaler"]
label_encoder = model_data["label_encoder"]
features = model_data["features"]

n_clusters = getattr(model, "n_clusters", "N/A")
inertia = getattr(model, "inertia_", "N/A")


# ------------------------------------------------------------
# TOP METRICS CARD DISPLAY
# ------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">ALGORITHM</div>
        <div class="metric-value">K-Means</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">CLUSTERS (K)</div>
        <div class="metric-value">{n_clusters}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    inertia_display = (
        f"{inertia:,.2f}"
        if isinstance(inertia, (int, float, np.number))
        else str(inertia)
    )

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">INERTIA (WCSS)</div>
        <div class="metric-value">{inertia_display}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">LEARNING TYPE</div>
        <div class="metric-value">Unsupervised</div>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🎯 Cluster Prediction",
    "📂 Dataset Analytics & Batch Predict",
    "ℹ️ Model Centroids & Architecture"
])


# ============================================================
# TAB 1 - SINGLE PREDICTION
# ============================================================
with tab1:

    st.markdown(
        '<div class="section-title">🎯 Single Post Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-box">
        Enter engagement metrics for a Facebook Live post below. The trained
        K-Means pipeline will scale inputs and assign the post to its nearest cluster.
    </div>
    """, unsafe_allow_html=True)

    if features is not None:
        feature_list = list(features)
    else:
        feature_list = [
            "status_type", "num_reactions", "num_comments",
            "num_shares", "num_likes", "num_loves",
            "num_wows", "num_hahas", "num_sads", "num_angrys"
        ]

    input_data = {}
    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_list):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            if feature == "status_type":
                if label_encoder is not None:
                    classes = list(label_encoder.classes_)
                    input_data[feature] = st.selectbox("Status Type", classes)
                else:
                    input_data[feature] = st.selectbox("Status Type", ["photo", "video", "link", "status"])
            else:
                input_data[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=0.0,
                    value=10.0 if "reaction" in feature or "like" in feature else 0.0,
                    step=1.0
                )

    st.markdown("---")

    predict_button = st.button("🚀 Predict Cluster", use_container_width=True, type="primary")

    if predict_button:
        try:
            input_df = pd.DataFrame([input_data])

            # Preprocessing: Encoding Categoricals
            if "status_type" in input_df.columns and label_encoder is not None:
                try:
                    input_df["status_type"] = label_encoder.transform(input_df["status_type"])
                except ValueError:
                    st.error("The selected status type was not present during training.")
                    st.stop()
            elif "status_type" in input_df.columns and label_encoder is None:
                # Basic mapping fallback if label encoder is missing
                mapping = {"photo": 0, "video": 1, "link": 2, "status": 3}
                input_df["status_type"] = input_df["status_type"].map(mapping).fillna(0)

            if features is not None:
                input_df = input_df[features]

            # Scaling
            if scaler is not None:
                input_scaled = scaler.transform(input_df)
            else:
                input_scaled = input_df.values

            # Inference
            prediction = model.predict(input_scaled)[0]

            st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-label">Assigned Group</div>
                <div class="prediction-number">Cluster #{prediction}</div>
                <div class="prediction-label">Facebook Live Post Segment</div>
            </div>
            """, unsafe_allow_html=True)

            if hasattr(model, "transform"):
                distances = model.transform(input_scaled)[0]
                nearest_distance = np.min(distances)
                st.info(f"📍 Distance from assigned centroid: **{nearest_distance:.4f}**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ============================================================
# TAB 2 - DATASET ANALYTICS & INTERACTIVE EXPLORER
# ============================================================
with tab2:

    st.markdown(
        '<div class="section-title">📂 Upload & Analyze Dataset</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload Live.csv for batch analysis", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Dataset uploaded successfully: {len(df):,} records.")

            # Process dataframe for cluster predictions
            df_proc = df.copy()

            # Prepare data to fit model expectations
            if features is not None:
                eval_features = [f for f in features if f in df_proc.columns]
            else:
                eval_features = [c for c in df_proc.columns if df_proc[c].dtype in [np.int64, np.float64]]

            if "status_type" in df_proc.columns and label_encoder is not None:
                try:
                    df_proc["status_type_encoded"] = label_encoder.transform(df_proc["status_type"].astype(str))
                except Exception:
                    df_proc["status_type_encoded"] = 0
            
            # Predict clusters across batch dataset
            try:
                if features is not None:
                    X_eval = df_proc[features].copy()
                    if "status_type" in X_eval.columns and label_encoder is not None:
                        X_eval["status_type"] = label_encoder.transform(X_eval["status_type"].astype(str))
                else:
                    X_eval = df_proc.select_dtypes(include=np.number)

                if scaler is not None:
                    X_scaled = scaler.transform(X_eval)
                else:
                    X_scaled = X_eval.values

                df["Predicted_Cluster"] = model.predict(X_scaled)
                has_clusters = True
            except Exception as pe:
                st.warning(f"Could not automatically assign clusters to uploaded CSV: {pe}")
                has_clusters = False

            # Display Data Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Rows", f"{df.shape[0]:,}")
            c2.metric("Total Columns", f"{df.shape[1]:,}")
            c3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
            c4.metric("Clusters Detected", f"{df['Predicted_Cluster'].nunique()}" if has_clusters else "N/A")

            # Interactive Visualizations
            st.markdown("### 📊 Interactive Visualizations")

            if has_clusters:
                vcol1, vcol2 = st.columns(2)

                with vcol1:
                    st.markdown("#### Cluster Distribution")
                    cluster_counts = df["Predicted_Cluster"].value_counts().reset_index()
                    cluster_counts.columns = ["Cluster", "Count"]
                    fig_bar = px.bar(
                        cluster_counts, 
                        x="Cluster", 
                        y="Count", 
                        color="Cluster",
                        title="Posts per Cluster",
                        text_auto=True,
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                with vcol2:
                    st.markdown("#### Engagement Scatter Plot")
                    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                    default_x = "num_reactions" if "num_reactions" in numeric_cols else numeric_cols[0]
                    default_y = "num_comments" if "num_comments" in numeric_cols else numeric_cols[min(1, len(numeric_cols)-1)]

                    x_axis = st.selectbox("X-Axis Feature", numeric_cols, index=numeric_cols.index(default_x))
                    y_axis = st.selectbox("Y-Axis Feature", numeric_cols, index=numeric_cols.index(default_y))

                    fig_scatter = px.scatter(
                        df, 
                        x=x_axis, 
                        y=y_axis, 
                        color=df["Predicted_Cluster"].astype(str),
                        title=f"{x_axis} vs {y_axis} by Cluster",
                        hover_data=df.columns[:5]
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

                st.markdown("#### 📋 Cluster Characteristics Summary")
                cluster_profile = df.groupby("Predicted_Cluster")[numeric_cols].mean().round(2)
                st.dataframe(cluster_profile, use_container_width=True)

            # Preview Data Table
            st.markdown("### 🔎 Dataset Preview")
            st.dataframe(df.head(20), use_container_width=True)

            # Export Button
            if has_clusters:
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Dataset with Cluster Predictions",
                    data=csv_data,
                    file_name="facebook_live_clustered.csv",
                    mime="text/csv",
                    type="primary"
                )

        except Exception as e:
            st.error(f"Unable to analyze uploaded file: {e}")

    else:

        st.info("💡 Upload `Live.csv` to unlock batch predictions, scatter plots, and summary metrics.")


# ============================================================
# TAB 3 - MODEL INFORMATION & CENTROIDS
# ============================================================
with tab3:

    st.markdown(
        '<div class="section-title">ℹ️ Model Architecture & Centroids</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ### 🤖 K-Means Clustering Mechanics
    K-Means partitions observations into $K$ clusters by placing centroids in feature space and iteratively refining their locations to minimize Within-Cluster Sum of Squares (WCSS/Inertia).
    """)

    if hasattr(model, "cluster_centers_"):
        st.markdown("### 🎯 Cluster Centroid Coordinates")
        
        centroids = model.cluster_centers_
        feature_names = features if features is not None else [f"Feature {i+1}" for i in range(centroids.shape[1])]
        
        centroid_df = pd.DataFrame(centroids, columns=feature_names)
        centroid_df.index.name = "Cluster"

        st.dataframe(centroid_df.style.background_gradient(cmap="Blues"), use_container_width=True)

        st.markdown("#### Centroid Comparison Chart")
        
        # Melt dataframe for plot
        centroid_melted = centroid_df.reset_index().melt(id_vars="Cluster", var_name="Feature", value_name="Centroid Value")
        
        fig_centroids = px.bar(
            centroid_melted, 
            x="Feature", 
            y="Centroid Value", 
            color="Cluster",
            barmode="group",
            title="Centroid Values per Feature (Scaled Space)"
        )
        st.plotly_chart(fig_centroids, use_container_width=True)

    st.markdown("### ⚙️ Model Hyperparameters")
    
    model_info = {
        "Algorithm": "K-Means",
        "Number of Clusters (K)": n_clusters,
        "Random State": getattr(model, "random_state", "Default"),
        "Max Iterations": getattr(model, "max_iter", 300),
        "Inertia (WCSS)": f"{inertia:,.2f}" if isinstance(inertia, (int, float, np.number)) else str(inertia),
        "Scaler Fitted": "Yes" if scaler is not None else "No",
        "Label Encoder Fitted": "Yes" if label_encoder is not None else "No"
    }

    st.table(pd.DataFrame(model_info.items(), columns=["Parameter", "Value"]))


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("""
<div class="footer">
    <b>K-Means Clustering Analytics Dashboard</b><br>
    Machine Learning Portfolio Project • Built with Python, Streamlit & Plotly
</div>
""", unsafe_allow_html=True)