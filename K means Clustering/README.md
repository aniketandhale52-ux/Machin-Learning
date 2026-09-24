# 🌿 Iris Intelligent Cluster Suite

### High-Performance Botanical Pattern Recognition & Machine Learning Architecture

**Iris Intelligent Cluster Suite** is an interactive **Unsupervised Machine Learning** application that uses the **K-Means Clustering algorithm** to discover natural groupings in the classic Iris dataset.

The application provides an interactive Streamlit dashboard where users can enter Iris flower measurements and receive a cluster assignment along with centroid distance analysis and visualizations.

---

## 📌 Project Overview

The project analyzes four physical measurements of Iris flowers:

* 🌿 Sepal Length
* 🌿 Sepal Width
* 🌷 Petal Length
* 🌷 Petal Width

The input features are standardized using `StandardScaler` and passed to a trained **K-Means model with K = 3**.

The application then displays:

* Predicted cluster
* Euclidean distance to the assigned centroid
* Cluster centroid profiles
* Petal-dimension cluster visualization
* Interactive dashboard

The project uses the classic **Iris Dataset** and performs clustering without depending on pre-existing class labels.

---

## 🎯 Objectives

1. Apply **K-Means Clustering** to Iris flower measurements.
2. Demonstrate an **Unsupervised Machine Learning** workflow.
3. Standardize numerical features before clustering.
4. Provide interactive cluster prediction.
5. Visualize cluster centroid characteristics.
6. Build a user-friendly ML dashboard using Streamlit.

---

## 🚀 Features

### 📊 Interactive Prediction

Users can enter:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

and execute cluster inference directly from the dashboard.

### 🤖 K-Means Clustering

The trained K-Means model uses:

```text
K = 3
```

to identify three morphological groups.

### 📏 Distance Analysis

The application calculates the Euclidean distance between the input observation and the cluster centroids.

### 📈 Cluster Visualizations

The dashboard provides:

* Cluster centroid radar profiles
* Petal dimension scatter plot
* User input visualization
* Cluster distribution analysis

### 🎨 Interactive UI

The Streamlit interface includes:

* Dark glassmorphism design
* Responsive layout
* Interactive tabs
* Styled prediction cards
* Plotly visualizations

---

## 🔄 Machine Learning Workflow

```text
Iris Dataset
     │
     ▼
Feature Selection
     │
     ▼
Sepal & Petal Measurements
     │
     ▼
StandardScaler
     │
     ▼
Trained K-Means Model
     │
     ▼
K = 3 Clusters
     │
     ▼
Cluster Prediction
     │
     ▼
Centroid Distance
     │
     ▼
Interactive Visualization
```

---

## 🧠 Algorithm

### K-Means Clustering

K-Means is an unsupervised learning algorithm that divides observations into a predefined number of clusters.

In this project:

```text
Number of Clusters = 3
```

The model assigns each input observation to the nearest cluster centroid.

The application also calculates the distance between the input and the selected centroid.

---

## 📂 Project Structure

```text
Iris-KMeans-Clustering/
│
├── app.py
├── Iris_KMeans_Clustering_Corrected.ipynb
├── iris_kmeans_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

> The pickle file contains the trained K-Means model, scaler, and feature information required by the Streamlit application.

---

## 🛠️ Technology Stack

### Programming Language

* Python

### Data Science

* NumPy
* Pandas
* Scikit-Learn

### Machine Learning

* K-Means Clustering
* StandardScaler

### Visualization

* Plotly
* Plotly Express

### Web Application

* Streamlit

### Model Storage

* Python Pickle

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/aniketandhale52-ux/Iris-KMeans-Clustering.git
```

Move into the project directory:

```bash
cd Iris-KMeans-Clustering
```

---

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the required packages:

```bash
pip install streamlit pandas numpy scikit-learn plotly
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

Usually, Streamlit runs at:

```text
http://localhost:8501
```

---

## 📊 Input Features

| Feature      | Description         | Unit |
| ------------ | ------------------- | ---- |
| Sepal Length | Length of the sepal | cm   |
| Sepal Width  | Width of the sepal  | cm   |
| Petal Length | Length of the petal | cm   |
| Petal Width  | Width of the petal  | cm   |

The application accepts values from **0 to 10 cm** for each measurement.

---

## 📈 Dashboard Output

After entering the measurements and clicking **Execute Cluster Inference**, the dashboard displays:

### Cluster Assignment

```text
Cluster X
```

### Centroid Distance

```text
Euclidean Distance to Centroid
```

### Input Metrics

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

### Visualizations

#### 1. Cluster Centroid Profiles

A radar chart shows the feature profiles of the different cluster centroids.

#### 2. Petal Dimensions vs K-Means Clusters

A scatter plot displays petal length and petal width for the Iris dataset and highlights the user's input.

---

## 📖 About the Dataset

The project uses the classic **Iris Dataset** available through Scikit-Learn.

The dataset contains four numerical measurements:

```text
Sepal Length
Sepal Width
Petal Length
Petal Width
```

The application loads the dataset using:

```python
from sklearn.datasets import load_iris
```

---

## 💡 Why Unsupervised Learning?

Unsupervised learning can identify patterns and structural groupings in data without using predefined class labels.

For this project, K-Means helps explore similarities between Iris observations based on their physical measurements.

Potential advantages include:

* Pattern discovery
* Automated grouping
* Numerical feature analysis
* Centroid-based similarity analysis
* Visualization of data structure

---

## 🎨 User Interface

The application contains two main sections:

### 📊 Prediction Form

Used for entering flower measurements and performing cluster inference.

### ℹ️ About Us

Contains:

* Project information
* Workflow
* Features
* Technology stack
* Clustering benefits
* Developer information
* Contact details

---

## 👨‍💻 Developer

### Aniket Andhale

**Role:** Machine Learning Developer

**Skills:**

* Python
* Machine Learning
* Scikit-Learn
* Pandas
* NumPy
* Streamlit

### Contact

📧 **Email:** [aniketandhale30@gmail.com](mailto:aniketandhale30@gmail.com)

💼 **LinkedIn:** [Aniket Andhale](https://www.linkedin.com/in/aniket-andhale-b072b8380)

🐙 **GitHub:** [aniketandhale52-ux](https://github.com/aniketandhale52-ux)

---

## 🔮 Future Improvements

Possible future enhancements include:

* Automatic cluster-to-species interpretation
* 3D cluster visualization
* PCA visualization
* Model evaluation metrics
* Dataset upload functionality
* Multiple prediction support
* Downloadable prediction reports
* Deployment on Streamlit Cloud

---

## 📜 License

This project is created for **educational and machine learning demonstration purposes**.

---

## ⭐ Project Summary

**Iris Intelligent Cluster Suite** demonstrates how an unsupervised machine learning algorithm can be combined with an interactive web dashboard to analyze and visualize patterns in botanical data.

```text
Python + Scikit-Learn + K-Means + Streamlit + Plotly
```

🌿 **Built with Python, Streamlit & Machine Learning**

**© 2026 Aniket Andhale**
