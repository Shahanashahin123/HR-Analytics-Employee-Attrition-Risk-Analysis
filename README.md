# HR Analytics – Employee Attrition Risk Analysis

## 📌 Project Overview

This project analyzes employee attrition using structured HR data to identify key factors contributing to workforce turnover. It combines exploratory data analysis with predictive modeling to support HR teams in designing proactive retention strategies.

The system provides risk probability scoring and categorizes employees into risk segments for better workforce planning.

---

## 🎯 Business Objective

- Identify drivers of employee attrition  
- Analyze trends across departments and experience levels  
- Predict employee attrition probability  
- Classify employees into risk categories  
- Support HR decision-making through data insights  

---

## 📊 Dataset

- IBM HR Analytics Dataset  
- Total Records: 1470  
- Target Variable: Attrition (Yes/No)

---

## 🔎 Analysis Performed

### 1️⃣ Data Cleaning
- Removed irrelevant features  
- Handled categorical encoding  
- Dropped duplicates  

### 2️⃣ Exploratory Data Analysis
- Attrition distribution analysis  
- Correlation heatmap  
- Department and role-based attrition patterns  

### 3️⃣ Predictive Modeling
- Random Forest Classifier  
- Class-balanced training  
- Model evaluation using confusion matrix and classification report  

### 4️⃣ Attrition Risk Scoring
- Generated attrition probability scores  
- Categorized employees into:
  - High Risk (>0.6)
  - Medium Risk (0.3–0.6)
  - Low Risk (<0.3)

---

## 📈 Key Insights

- Overtime strongly influences attrition probability  
- Lower monthly income correlates with higher turnover  
- Specific job roles show higher attrition patterns  
- Risk scoring enables proactive workforce retention planning  

---

## 🛠 Tools & Technologies

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
python hr_attrition_analysis.py
