# 🚗 Auto Price Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting the **price of automobiles** using Machine Learning regression techniques.

The project uses an automobile dataset and applies data preprocessing, feature selection, model training, and performance evaluation. Two regression algorithms are explored:

* 🌳 Decision Tree Regressor
* 🌲 Random Forest Regressor

The Random Forest model is finally trained and saved as a `.pkl` file for future use.

---

## 🎯 Objectives

The main objectives of this project are:

* To understand and preprocess automobile data.
* To handle missing and invalid values.
* To select relevant features for price prediction.
* To train regression models for automobile price prediction.
* To evaluate model performance using different regression metrics.
* To save the trained Random Forest model for future predictions.

---

## 📂 Dataset

The project uses:

`autos_dataset.csv`

The dataset contains various automobile-related attributes such as:

* Symboling
* Wheel Base
* Length
* Width
* Height
* Curb Weight
* Engine Size
* Number of Cylinders
* Compression Ratio
* Horsepower
* Peak RPM
* City MPG
* Highway MPG
* Price

The target variable is:

**Price**

---

## 🛠️ Technologies Used

| Technology       | Purpose                 |
| ---------------- | ----------------------- |
| Python           | Programming Language    |
| NumPy            | Numerical Computation   |
| Pandas           | Data Processing         |
| Scikit-learn     | Machine Learning        |
| Matplotlib       | Data Visualization      |
| Seaborn          | Visualization           |
| Pickle           | Model Serialization     |
| Jupyter Notebook | Development Environment |

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Data Cleaning
   ↓
Handle Missing Values
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Decision Tree Regression
   ↓
Random Forest Regression
   ↓
Model Evaluation
   ↓
Save Random Forest Model
```

---

## 🧹 Data Preprocessing

Several preprocessing operations were performed on the dataset.

### 1. Load Dataset

```python
df = pd.read_csv('autos_dataset.csv')
```

### 2. Convert Number of Cylinders

Categorical cylinder values were converted into numerical values.

For example:

```text
two    → 2
three  → 3
four   → 4
five   → 5
six    → 6
eight  → 8
twelve → 12
```

### 3. Handle Invalid Values

The `?` values were replaced with `NaN`.

```python
df.replace({'?': np.nan}, inplace=True)
```

### 4. Remove Unnecessary Features

The following columns were removed:

```text
make
fuel-type
aspiration
engine-type
normalized-losses
num-of-doors
bore
stroke
fuel-system
body-style
drive-wheels
engine-location
```

### 5. Convert Data Types

The following columns were converted into floating-point values:

```text
price
peak-rpm
horsepower
```

### 6. Handle Missing Values

Missing values in:

* Price
* Peak RPM
* Horsepower

were replaced using the respective column mean.

---

## 📊 Feature and Target Selection

The target variable is:

```python
y = df['price']
```

The remaining selected columns are used as input features:

```python
X = df.drop(['price'], axis=1)
```

---

## ✂️ Train-Test Split

The dataset was divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Split Ratio

* **80% → Training Data**
* **20% → Testing Data**

---

# 🌳 Decision Tree Regression

A Decision Tree Regressor was trained on the automobile dataset.

```python
dt_reg = DecisionTreeRegressor()
dt_reg.fit(X_train, y_train)
```

The model was evaluated on both training and testing datasets.

---

# 🌲 Random Forest Regression

A Random Forest Regressor was also implemented.

```python
rf_reg = RandomForestRegressor(
    random_state=42
)

rf_reg.fit(X_train, y_train)
```

Random Forest combines multiple decision trees to improve prediction performance and reduce overfitting compared with a single decision tree.

---

## 📈 Model Evaluation

The models are evaluated using the following regression metrics:

### Mean Squared Error (MSE)

Measures the average squared difference between actual and predicted values.

### Root Mean Squared Error (RMSE)

RMSE is the square root of MSE and represents prediction error in the same unit as the target variable.

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted values.

### R² Score

R² indicates how well the model explains the variation in the target variable.

Higher R² generally indicates better predictive performance.

---

## 🔧 Hyperparameter Tuning

Grid Search was explored for Decision Tree optimization using parameters such as:

```text
splitter
max_depth
min_samples_leaf
max_features
max_leaf_nodes
```

The model was configured with:

```python
GridSearchCV(
    estimator=dt_reg,
    param_grid=parameters,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=3
)
```

---

## 💾 Model Saving

The trained Random Forest model was saved using Pickle.

```python
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_reg, f)
```

The feature names were also saved:

```python
with open("features.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)
```

### Generated Files

```text
random_forest_model.pkl
features.pkl
```

These files can be used later to load the trained model and make predictions on new automobile data.

---

## 📁 Project Structure

```text
Auto-Price-Prediction/
│
├── autos_dataset.csv
├── Auto_Price_Prediction.ipynb
├── random_forest_model.pkl
├── features.pkl
├── README.md
└── requirements.txt
```

---

## ▶️ How to Run the Project

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
```

### Step 2: Navigate to the Project

```bash
cd Auto-Price-Prediction
```

### Step 3: Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Step 4: Start Jupyter Notebook

```bash
jupyter notebook
```

### Step 5: Open the Notebook

Open:

```text
Auto_Price_Prediction.ipynb
```

Run the cells sequentially.

---

## 📦 Requirements

```text
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyter
```

---

## 🚀 Future Improvements

The project can be further improved by:

* Developing a web-based prediction application using Streamlit.
* Adding interactive visualizations.
* Performing advanced feature engineering.
* Comparing additional regression algorithms.
* Performing more extensive hyperparameter optimization.
* Adding prediction input forms for users.
* Deploying the trained model as a web application.
* Adding model performance comparison charts.

---

## 🏆 Key Learning Outcomes

Through this project, the following concepts were implemented:

* Data Loading
* Data Cleaning
* Missing Value Handling
* Feature Selection
* Train-Test Split
* Decision Tree Regression
* Random Forest Regression
* Hyperparameter Tuning
* Model Evaluation
* Model Serialization using Pickle

---

## 👨‍💻 Author

**Aniket Andhale**

B.Tech – Artificial Intelligence & Data Science

---

## ⭐ Conclusion

This project demonstrates how Machine Learning regression algorithms can be applied to automobile data to predict vehicle prices.

Both **Decision Tree Regression** and **Random Forest Regression** were implemented and evaluated using standard regression metrics. The trained Random Forest model was saved using Pickle so that it can be reused for future automobile price predictions.

If you find this project useful, consider giving the repository a ⭐ on GitHub.
