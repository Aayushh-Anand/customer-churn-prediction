from pathlib import Path
import joblib
import pandas as pd


# Paths

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "churn_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
CONFIG_PATH = MODEL_DIR / "model_config.pkl"


# Load trained model and preprocessing objects

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
config = joblib.load(CONFIG_PATH)

FEATURES = config["features"]
THRESHOLD = config["threshold"]


# Prediction Function

def predict_churn(customer_data: dict) -> dict:
    """
    Predict customer churn probability and risk.

    Parameters
    ----------
    customer_data : dict
        Customer information containing all required features.

    Returns
    -------
    dict
        Churn probability, prediction and risk level.
    """

    # Convert input dictionary into DataFrame
    customer_df = pd.DataFrame(
        [customer_data],
        columns=FEATURES
    )

    # Preprocess customer data
    processed_data = preprocessor.transform(customer_df)

    # Get churn probability
    churn_probability = model.predict_proba(
        processed_data
    )[0][1]

    # Apply saved threshold
    prediction = int(
        churn_probability >= THRESHOLD
    )

    # Determine risk level
    if churn_probability >= 0.70:
        risk = "High"
    elif churn_probability >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "churn_probability": round(
            float(churn_probability),
            4
        ),
        "prediction": prediction,
        "prediction_label": (
            "Likely to Churn"
            if prediction == 1
            else "Likely to Stay"
        ),
        "risk_level": risk,
        "threshold": THRESHOLD
    }


# Example Prediction

if __name__ == "__main__":

    sample_customer = {
        "Gender": "Male",
        "Senior Citizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "Tenure Months": 8,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 85.50,
        "Total Charges": 684.00,
        "CLTV": 3500
    }

    result = predict_churn(sample_customer)

    print("\nCustomer Churn Prediction")
    print("-" * 35)

    print(
        f"Churn Probability: "
        f"{result['churn_probability']:.2%}"
    )

    print(
        f"Prediction: "
        f"{result['prediction_label']}"
    )

    print(
        f"Risk Level: "
        f"{result['risk_level']}"
    )

    print(
        f"Threshold: "
        f"{result['threshold']:.2f}"
    )