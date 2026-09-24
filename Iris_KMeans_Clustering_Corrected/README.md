# 🌸 IrisCluster AI

### Intelligent Iris Flower Clustering using K-Means

IrisCluster AI is a Machine Learning project that uses **K-Means Clustering**, an unsupervised learning algorithm, to group similar Iris flowers based on their physical measurements.

The project includes an interactive **Streamlit web application** where users can explore the Iris dataset, visualize clusters, and predict the cluster of a new flower.

---

## 📌 Project Overview

The Iris dataset contains measurements of Iris flowers using four features:

* 🌱 Sepal Length
* 🌱 Sepal Width
* 🌸 Petal Length
* 🌸 Petal Width

The project applies **StandardScaler** for feature scaling and **K-Means Clustering** to divide the flowers into **3 clusters**.

### Machine Learning Type

**Unsupervised Machine Learning**

### Algorithm

**K-Means Clustering**

### Number of Clusters

**K = 3**

---

## 🎯 Objectives

* Analyze the Iris flower dataset.
* Preprocess and standardize numerical features.
* Determine suitable clusters using clustering evaluation techniques.
* Apply K-Means clustering.
* Visualize the resulting clusters.
* Build an interactive Streamlit application.
* Predict the cluster of a new Iris flower based on its measurements.

---

## 🧠 Features

### 🔮 Cluster Prediction

Enter the four flower measurements and the application predicts the corresponding K-Means cluster.

### 📊 Dataset Explorer

View the Iris dataset and explore feature distributions.

### 📈 Cluster Visualization

Visualize K-Means clusters using **PCA (Principal Component Analysis)**.

### 📋 Model Information

View information about the trained K-Means model, number of clusters, and model inertia.

### 🎨 Interactive Dashboard

A simple and attractive Streamlit interface provides easy navigation between different project sections.

---

## 🛠️ Technologies Used

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Programming Language      |
| Pandas       | Data Processing           |
| NumPy        | Numerical Computing       |
| Scikit-learn | Machine Learning          |
| Plotly       | Interactive Visualization |
| Streamlit    | Web Application           |
| Pickle       | Model Serialization       |
| PCA          | Cluster Visualization     |

---

## 📂 Project Structure

```text
IrisCluster-AI/
│
├── app.py
├── iris_kmeans_model.pkl
├── Iris.csv
├── README.md
│
└── assets/
    └── iris_background.jpg
```

> `Iris.csv` and `assets/iris_background.jpg` are optional depending on the version of the project. The Streamlit application can use the built-in Iris dataset.

---

## ⚙️ Machine Learning Workflow

```text
Iris Dataset
      ↓
Data Preprocessing
      ↓
Feature Selection
      ↓
StandardScaler
      ↓
K-Means Clustering
      ↓
K = 3
      ↓
Cluster Analysis
      ↓
PCA Visualization
      ↓
Streamlit Application
```

---

## 📊 Input Features

The model uses four numerical features:

```text
SepalLengthCm
SepalWidthCm
PetalLengthCm
PetalWidthCm
```

Example:

| Feature      | Example |
| ------------ | ------: |
| Sepal Length |  5.1 cm |
| Sepal Width  |  3.5 cm |
| Petal Length |  1.4 cm |
| Petal Width  |  0.2 cm |

---

## 🤖 Model

The final model uses:

```text
Algorithm: K-Means Clustering
Number of Clusters: 3
Initialization: k-means++
n_init: 10
Random State: 42
```

The features are standardized before being provided to the K-Means model.

---

## 💾 Model Serialization

The trained model, scaler, and feature information are stored in:

```text
iris_kmeans_model.pkl
```

The pickle file contains:

```python
{
    "model": kmeans,
    "scaler": scaler,
    "features": features
}
```

This allows the Streamlit application to load the trained model without training it again.

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

### 2. Open the Project

```bash
cd IrisCluster-AI
```

### 3. Install Required Libraries

```bash
pip install streamlit pandas numpy scikit-learn plotly
```

---

## ▶️ Run the Application

Run the following command in the project folder:

```bash
python -m streamlit run app.py
```

Or:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🖥️ Application Pages

### 🏠 Dashboard

Provides an overview of:

* Iris dataset
* Number of samples
* Number of features
* Number of clusters
* Species distribution
* Cluster distribution

### 🔮 Predict Cluster

Users can enter:

```text
Sepal Length
Sepal Width
Petal Length
Petal Width
```

The application then predicts the K-Means cluster.

### 📊 Explore Data

Allows users to:

* View the dataset
* Select features
* Analyze feature distributions
* Compare measurements between species

### 📈 Cluster Visualization

Uses PCA to reduce the four-dimensional feature space into two dimensions and displays the K-Means clusters visually.

### ℹ️ About

Provides information about:

* Project
* Algorithm
* Dataset
* Features
* Technologies

---

## 📌 Example Prediction

Input:

```text
Sepal Length : 5.1
Sepal Width  : 3.5
Petal Length : 1.4
Petal Width  : 0.2
```

The application returns:

```text
Prediction Result
Cluster X
```

> Cluster numbers are machine-generated labels and do not directly represent Iris species names.

---

## 📈 Visualization

The project uses:

* Bar Charts
* Pie Charts
* Histograms
* Box Plots
* PCA Scatter Plot

These visualizations help understand the dataset and clustering results.

---

## 🌟 Advantages

* Simple and easy-to-use interface.
* Uses an unsupervised Machine Learning algorithm.
* Interactive cluster prediction.
* Interactive visualizations.
* Trained model can be reused using Pickle.
* PCA provides a clear visual representation of clusters.
* Suitable for Machine Learning beginners and academic demonstrations.

---

## 🔮 Future Enhancements

Possible future improvements include:

* Upload custom CSV datasets.
* Automatic selection of optimal K.
* Silhouette score visualization.
* 3D cluster visualization.
* Model performance dashboard.
* Download prediction results.
* Add more clustering algorithms such as DBSCAN and Hierarchical Clustering.
* Deploy the application online.

---

## 📚 Dataset

The project uses the **Iris flower dataset**, containing 150 samples with four numerical measurements.

The four measurements are:

```text
Sepal Length
Sepal Width
Petal Length
Petal Width
```

The dataset contains three known Iris species:

```text
Setosa
Versicolor
Virginica
```

The species labels are used for dataset exploration/evaluation, while the K-Means algorithm itself performs clustering without using species labels during training.

---

## 👨‍💻 Author

**Aniket Andhale**

B.Tech – Artificial Intelligence

---

## ⭐ Project Highlights

```text
🌸 IrisCluster AI
🤖 K-Means Clustering
📊 4 Input Features
🎯 3 Clusters
📈 PCA Visualization
🔮 Cluster Prediction
🖥️ Streamlit Dashboard
🐍 Python Machine Learning
```

---

## 📜 License

This project is created for **educational and academic purposes**.
