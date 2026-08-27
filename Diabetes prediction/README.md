# 🩸 DiabetesAI - Risk Intelligence Platform

An AI-powered diabetes risk prediction platform that uses a **Decision Tree Machine Learning model** to estimate diabetes risk based on clinical and physiological parameters.

🌐 **Live Website:** https://anysdibetiesprediction.netlify.app/

---

## 📌 Project Overview

**DiabetesAI** is a machine learning-based web application designed for educational and analytical demonstration of diabetes risk prediction.

The system takes 8 clinical parameters as input and uses a trained **Decision Tree Classifier** to generate:

- Diabetes prediction
- Diabetes probability
- Risk category
- Classification confidence
- Patient metric analysis
- Prediction history
- Dataset exploration
- Model architecture information

The original machine learning project was developed using Python and Scikit-learn. The web version has been converted into a **static HTML, CSS and JavaScript application** so it can be deployed directly on platforms such as Netlify.

---

## 🌐 Live Demo

### 🚀 DiabetesAI Web Application

👉 **https://anysdibetiesprediction.netlify.app/**

You can access the complete prediction platform directly from the link above.

---

## ✨ Features

### 🔮 Prediction Engine

Enter the following patient parameters:

1. Pregnancies
2. Glucose
3. Blood Pressure
4. Skin Thickness
5. Insulin
6. BMI
7. Diabetes Pedigree Function
8. Age

The system generates a diabetes risk prediction using the trained Decision Tree model.

---

### 📊 Risk Analysis

The application provides:

- Diabetes probability percentage
- Classification result
- Risk category
- Classification confidence
- Risk probability meter
- Patient metric visualization
- Contributory factor insights

### Risk Categories

| Probability | Risk Level |
|---|---|
| 0% - 30% | 🟢 Low Risk |
| 30% - 60% | 🟡 Moderate Risk |
| 60% - 80% | 🟠 High Risk |
| 80% - 100% | 🔴 Very High Risk |

---

## 📜 Prediction History

The application stores prediction records locally in the browser using:

```text
localStorage
