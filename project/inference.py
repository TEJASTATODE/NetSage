import numpy as np
import pandas as pd
import joblib
import shap

from tensorflow.keras.models import load_model

# =========================================================
# LOAD ARTIFACTS
# =========================================================

print("Loading NetSage Models...")

autoencoder = load_model("model.keras")

scaler = joblib.load("scaler.pkl")

columns = joblib.load("columns.pkl")

xgb_model = joblib.load("xgb_model.pkl")

feature_names = joblib.load(
    "xgb_feature_names.pkl"
)

config = joblib.load(
    "inference_config.pkl"
)

# =========================================================
# THRESHOLD
# =========================================================

FINAL_THRESHOLD = config[
    "final_threshold"
]

# =========================================================
# SHAP EXPLAINER
# =========================================================

explainer = shap.TreeExplainer(xgb_model)

print("Models Loaded Successfully")

def predict_anomaly(packet: dict):

    try:


        packet_df = pd.DataFrame([packet])

        for col in columns:

            if col not in packet_df.columns:

                packet_df[col] = 0

        packet_df = packet_df[columns]



        packet_df = packet_df.apply(
            pd.to_numeric,
            errors="coerce"
        )

        packet_df = packet_df.fillna(0)


        X = scaler.transform(packet_df)


        reconstructed = autoencoder.predict(
            X,
            verbose=0
        )

        reconstruction_error = float(
            np.mean(
                np.square(X - reconstructed)
            )
        )

        X_hybrid = np.hstack([
            X,
            np.array([
                [reconstruction_error]
            ])
        ])

        xgb_probability = float(
            xgb_model.predict_proba(X_hybrid)[0][1]
        )

        xgb_prediction = int(
            xgb_model.predict(X_hybrid)[0]
        )


        shap_values = explainer.shap_values(
            X_hybrid
        )[0]

        top_indices = np.argsort(
            np.abs(shap_values)
        )[-5:][::-1]

        top_features = []

        for idx in top_indices:

            top_features.append({

                "feature": feature_names[idx],

                "importance": float(
                    shap_values[idx]
                )
            })


        if xgb_probability > 0.90:

            severity = "CRITICAL"

        elif xgb_probability > 0.75:

            severity = "HIGH"

        elif xgb_probability > 0.50:

            severity = "MEDIUM"

        else:

            severity = "LOW"

        # =============================================
        # FINAL RESPONSE
        # =============================================

        return {

            "anomaly_score": reconstruction_error,

            "threshold": FINAL_THRESHOLD,

            "xgb_probability": xgb_probability,

            "is_anomaly": bool(xgb_prediction),

            "severity": severity,

            "top_features": top_features
        }

    except Exception as e:

        return {
            "error": str(e)
        }