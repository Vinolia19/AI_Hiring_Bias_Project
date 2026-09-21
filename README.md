# AI Hiring Bias Detection Using Machine Learning and Fairness Analysis

## 📌 Project Overview

This project investigates whether a simulated AI-assisted hiring system produces systematically different evaluations or hiring decisions across candidate groups.

The project combines exploratory data analysis, machine learning, statistical analysis, and fairness metrics to examine patterns in AI-assisted hiring outcomes.

## 🎯 Research Question

> Does the simulated AI hiring system produce systematically different evaluations or decisions across candidate groups, and which candidate characteristics are associated with those outcomes?

## 🧠 Objectives

- Analyze candidate and hiring data
- Explore differences in AI-generated candidate scores
- Examine hiring outcomes across candidate groups
- Measure fairness-related disparities
- Build machine learning models for hiring prediction
- Identify features associated with model predictions
- Present findings through an interactive Streamlit dashboard

## 📊 Dataset

The project uses a simulated dataset containing 5,000 candidate records.

Example features include:

- Age
- Gender
- Education level
- University tier
- Years of experience
- Employment gap
- Technical skill score
- Communication score
- Aptitude test score
- Coding test score
- Project count
- GitHub activity
- Certifications
- Expected salary
- AI resume score
- AI bias score
- Hiring outcome

## 🔬 Methodology

The project follows these stages:

1. Data loading and preprocessing
2. Exploratory Data Analysis (EDA)
3. Group-based hiring analysis
4. Fairness analysis
5. Statistical analysis
6. Logistic Regression
7. Random Forest
8. Model evaluation
9. Feature importance analysis
10. Interactive dashboard development

## ⚖️ Fairness Analysis

The project examines fairness-related measures such as:

- Selection rate
- Demographic parity difference
- Demographic parity ratio
- Hiring rate differences between groups
- Differences in AI-generated scores

These metrics describe observed disparities in the simulated dataset. They do not by themselves establish the cause of a disparity or prove discrimination.

## 🤖 Machine Learning Models

Two machine learning models were evaluated:

- Logistic Regression
- Random Forest

The models were trained to predict the simulated hiring outcome using candidate characteristics.

Model performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## 🖥️ Interactive Dashboard

The project includes a Streamlit dashboard with:

- Project Overview
- Dataset Analysis
- Fairness Analysis
- ML Model Results
- Feature Importance
- Methodology
- Conclusion

## 🚀 Live Demo

**Streamlit App:**

https://ai-hiring-bias-detection.streamlit.app

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Fairlearn
- Streamlit
- Jupyter Notebook
- Git & GitHub

## 📁 Project Structure

```text
AI_Hiring_Bias_Project/
│
├── AI_Hiring_Bias_Dataset.csv
├── aiapp.py
├── requirements.txt
├── README.md
│
├── notebooks/
│   └── ai_hiring_bias.ipynb
│
├── results/
│   ├── Final_ML_Summary.csv
│   ├── Final_Fairness_Summary.csv
│   ├── Gender_AI_Fairness_Analysis.csv
│   ├── Model_Predictions.csv
│   ├── ML_Model_Comparison.csv
│   └── Random_Forest_Feature_Importance.csv
│
└── models/