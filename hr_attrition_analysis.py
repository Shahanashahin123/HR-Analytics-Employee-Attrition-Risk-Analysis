# HR Analytics: Employee Attrition Prediction
# Mini Project - Beginner Friendly

# ------------------------------
# Step 1: Import Libraries
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib  # To save the model

# ------------------------------
# Step 2: Load Dataset
# ------------------------------
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
print("First 5 rows of dataset:")
print(df.head())

# ------------------------------
# Step 3: Data Cleaning
# ------------------------------
# Check info
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# Drop irrelevant columns
columns_to_drop = ['EmployeeNumber', 'StandardHours', 'Over18', 'EmployeeCount']
for col in columns_to_drop:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Convert target column to numeric
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

print("\nData after cleaning:")
print(df.head())
print("Data shape:", df.shape)

# ------------------------------
# Step 4: Exploratory Data Analysis (EDA)
# ------------------------------
# Attrition count
plt.figure(figsize=(6,4))
sns.countplot(x='Attrition', data=df)
plt.title("Attrition Count")
plt.savefig("eda_attrition_count.png")
plt.show()

# Correlation matrix for numeric features
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
plt.figure(figsize=(12,10))
sns.heatmap(df[num_cols].corr(), cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.savefig("eda_corr_matrix.png")
plt.show()

# ------------------------------
# Step 5: Encode Categorical Variables
# ------------------------------
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns:", cat_cols)

# Binary columns
le = LabelEncoder()
binary_cols = [c for c in cat_cols if df[c].nunique() == 2]
multi_cols = [c for c in cat_cols if df[c].nunique() > 2]

# Encode binary columns
for c in binary_cols:
    df[c] = le.fit_transform(df[c].astype(str))

# One-hot encode multi-class columns
df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
print("Data shape after encoding:", df.shape)

# ------------------------------
# Step 6: Split Data into Train and Test Sets
# ------------------------------
y = df['Attrition']
X = df.drop('Attrition', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ------------------------------
# Step 7: Train RandomForest Model
# ------------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:,1]

# ------------------------------
# Step 8: Evaluate Model
# ------------------------------
print("\nRandom Forest Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# ------------------------------
# Step 9: Feature Importance
# ------------------------------
fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 20 Features:")
print(fi.head(20))

# Plot top 20 features
plt.figure(figsize=(8,6))
fi.head(20).sort_values().plot(kind='barh')
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.show()

# ------------------------------
# Step 10: Save Model
# ------------------------------
joblib.dump(rf, "rf_attrition_model.pkl")
print("\nModel saved as rf_attrition_model.pkl")
