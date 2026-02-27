
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

print("\nChecking Missing Values:")
print(df.isnull().sum())

df = df.drop_duplicates()


columns_to_drop = ['EmployeeNumber', 'StandardHours', 'Over18', 'EmployeeCount']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)


df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

print("\nData after cleaning:")
print(df.head())




plt.figure(figsize=(6,4))
sns.countplot(x='Attrition', data=df)
plt.title("Employee Attrition Distribution")
plt.savefig("eda_attrition_count.png")
plt.show()

num_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(12,10))
sns.heatmap(df[num_cols].corr(), cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.savefig("eda_corr_matrix.png")
plt.show()


cat_cols = df.select_dtypes(include=['object']).columns

le = LabelEncoder()

for col in cat_cols:
    if df[col].nunique() == 2:
        df[col] = le.fit_transform(df[col])

df = pd.get_dummies(df, drop_first=True)

print("\nData shape after encoding:", df.shape)

X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]


print("\nModel Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.show()


feature_importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 20 Important Features:")
print(feature_importance.head(20))

plt.figure(figsize=(8,6))
feature_importance.head(20).sort_values().plot(kind='barh')
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.show()


risk_df = X_test.copy()
risk_df['Actual_Attrition'] = y_test.values
risk_df['Predicted_Attrition'] = y_pred
risk_df['Attrition_Risk_Probability'] = y_proba

def risk_category(prob):
    if prob > 0.6:
        return "High Risk"
    elif prob > 0.3:
        return "Medium Risk"
    else:
        return "Low Risk"

risk_df['Risk_Category'] = risk_df['Attrition_Risk_Probability'].apply(risk_category)

print("\nSample Risk Categorization:")
print(risk_df[['Attrition_Risk_Probability', 'Risk_Category']].head())


print("\n==============================")
print("Business Insights Summary")
print("==============================")
print("1. Overtime shows strong influence on employee attrition.")
print("2. Lower monthly income correlates with higher attrition risk.")
print("3. Certain job roles demonstrate higher turnover probability.")
print("4. Risk scoring enables proactive retention strategies.")
print("==============================")


joblib.dump(rf, "rf_attrition_model.pkl")
print("\nModel saved successfully as rf_attrition_model.pkl")
