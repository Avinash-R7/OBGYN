import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.abspath("src"))

from explainability import explain_prediction

model = joblib.load("src/best_model.pkl")

importance_df = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nMaternal Health Risk Prediction (Terminal Mode)")
print("Press Ctrl+C anytime to stop.\n")

while True:
    try:
        age = float(input("Age: "))
        sbp = float(input("Systolic BP: "))
        dbp = float(input("Diastolic BP: "))
        bs = float(input("Blood Sugar: "))
        temp = float(input("Body Temperature: "))
        hr = float(input("Heart Rate: "))

        patient = pd.DataFrame(
            [[age, sbp, dbp, bs, temp, hr]],
            columns=[
                "Age",
                "SystolicBP",
                "DiastolicBP",
                "BS",
                "BodyTemp",
                "HeartRate"
            ]
        )

        probs = model.predict_proba(patient)[0]
        risk = ["LOW", "MID", "HIGH"][probs.argmax()]

        explanation = explain_prediction(model, patient.iloc[0], importance_df)

        print("\nPrediction Result")
        print("------------------")
        print("Risk Level:", risk)
        print("Explanation:", explanation)
        print("\nEnter another patient data...\n")

    except KeyboardInterrupt:
        print("\nProgram stopped.")
        break