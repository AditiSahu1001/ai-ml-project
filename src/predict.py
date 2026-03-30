import joblib
import pandas as pd

# Load saved files
model = joblib.load("model/hypertension_model.pkl")
encoders = joblib.load("model/encoders.pkl")
columns = joblib.load("model/columns.pkl")

print("====== Hypertension Risk Prediction System ======\n")

input_data = {}

# Take input for each column
for col in columns:
    if col in encoders:  # categorical column
        le = encoders[col]
        print(f"{col} (Options: {list(le.classes_)})")
        value = input(f"Enter {col}: ")
        try:
            value = le.transform([value])[0]
        except:
            print(f"Invalid value for {col}")
            exit()
    else:  # numerical column
        value = float(input(f"Enter {col}: "))
    
    input_data[col] = value

# Create dataframe
input_df = pd.DataFrame([input_data])

# Prediction
prediction = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]

# Output
print("\n===== RESULT =====")
if prediction == 1:
    print("Hypertension Risk: HIGH")
else:
    print("Hypertension Risk: LOW")

print(f"Probability: {round(prob*100, 2)}%\nThis system is an academic machine learning project using synthetic Kaggle data — it is for educational purposes only and not a medical tool.")
