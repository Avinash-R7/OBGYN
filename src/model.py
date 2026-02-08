import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_model(data_path):
    # Load processed dataset
    df = pd.read_csv(data_path)

    X = df.drop("RiskLabel", axis=1)
    y = df["RiskLabel"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train final model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "src/best_model.pkl")

    # Feature importance (needed for explanation)
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return model, importance_df, X_test

if __name__ == "__main__":
    model, importance_df, _ = train_model(
        "data/processed/final_maternal_preterm_risk.csv"
    )
    importance_df.to_csv("src/feature_importance.csv", index=False)
    print("Model and feature importance saved successfully.")
