# IMPORTS 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

#  MAIN FUNCTION 
def main():
    print(" Loading dataset")
    df = pd.read_csv("data/hypertension_dataset.csv")

    #  DATA PREPROCESSING 
    df['Medication'] = df['Medication'].fillna('None')
    df['Has_Hypertension'] = df['Has_Hypertension'].map({'Yes': 1, 'No': 0})

    categorical_cols = ['BP_History', 'Medication', 'Family_History',
                        'Exercise_Level', 'Smoking_Status']
    
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    X = df.drop('Has_Hypertension',axis=1)
    y = df['Has_Hypertension']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #  MODEL TRAINING 
    print("\n Training models...\n")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }

    results = []
    for name, model in models.items():
        if name in ["Logistic Regression", "KNN"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        })
        print(f"{name} done")

    results_df = pd.DataFrame(results)
    print("\nModel Comparison (sorted by F1):")
    print(results_df.sort_values(by="F1",ascending=False))

    # FINAL MODEL (XGBoost) =
    print("\nTraining FINAL model (XGBoost)")
    final_model = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6)
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]

    print("\n=== FINAL MODEL PERFORMANCE ===")
    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    print("F1 Score :", round(f1_score(y_test, y_pred), 4))
    print("ROC-AUC :", round(roc_auc_score(y_test, y_prob), 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # SAVE MODEL 
    os.makedirs("model", exist_ok=True)
    joblib.dump(final_model, "model/hypertension_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(le_dict, "model/encoders.pkl")
    joblib.dump(X.columns.tolist(), "model/columns.pkl")
    print("\n Model and Scaler saved successfully in 'model/' folder!")

    # = FEATURE IMPORTANCE 
    importances = pd.Series(final_model.feature_importances_, index=X.columns)
    print("\n Top 10 Important Features:")
    print(importances.sort_values(ascending=False).head(10))

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=importances.sort_values(ascending=False).head(10),
        y=importances.sort_values(ascending=False).head(10).index,
        palette="viridis"
    )
    plt.title("Top 10 Important Features ")
    plt.xlabel("Feature Importance Score")
    plt.tight_layout()
    plt.show()


#  RUN 
if __name__ == "__main__":
    main()
