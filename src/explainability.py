import numpy as np

NORMAL_RANGES = {
    "SystolicBP": (90, 120),
    "DiastolicBP": (60, 80),
    "BS": (4.0, 7.8),
    "HeartRate": (60, 100),
    "BodyTemp": (97.0, 99.0),
    "Age": (18, 35)
}

def explain_prediction(model, patient_row, importance_df):
    probs = model.predict_proba(patient_row.to_frame().T)[0]
    predicted_class = int(np.argmax(probs))

    risk_map = {0: "LOW RISK", 1: "MID RISK", 2: "HIGH RISK"}
    reasons = []

    for feature in importance_df["Feature"][:3]:
        value = patient_row[feature]
        low, high = NORMAL_RANGES.get(feature, (None, None))

        if low is not None and value > high:
            reasons.append(f"{feature} is high ({value})")
        elif low is not None and value < low:
            reasons.append(f"{feature} is low ({value})")

    if not reasons:
        reasons.append("most clinical indicators are within normal range")

    return (
        f"The model predicts {risk_map[predicted_class]} because "
        + ", ".join(reasons) + "."
    )
