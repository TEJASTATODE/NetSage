import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import shap

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score
)

from xgboost import XGBClassifier

import matplotlib.pyplot as plt

# =========================================================
# SEED
# =========================================================

np.random.seed(42)
tf.random.set_seed(42)

# =========================================================
# LOAD DATASET
# =========================================================

DATASET_PATH = (
    r"C:\Users\TEJAS\OneDrive\Desktop\PROJECTS\IDS\project\data\UNSW-NB15_1_with_features.csv"
)

df = pd.read_csv(
    DATASET_PATH,
    low_memory=False
)

print("Original Shape:", df.shape)

# =========================================================
# CLEAN LABEL
# =========================================================

df["Label"] = pd.to_numeric(
    df["Label"],
    errors="coerce"
)

print("\nLabel Distribution:")
print(df["Label"].value_counts())

# =========================================================
# SPLIT NORMAL / ATTACK
# =========================================================

normal_df = df[df["Label"] == 0]
attack_df = df[df["Label"] == 1]

print("\nNormal Shape:", normal_df.shape)
print("Attack Shape:", attack_df.shape)

# =========================================================
# REMOVE LABEL
# =========================================================

normal_df = normal_df.drop(columns=["Label"])
attack_df = attack_df.drop(columns=["Label"])

# =========================================================
# KEEP NUMERIC FEATURES ONLY
# =========================================================

normal_df = normal_df.select_dtypes(include=["number"])
attack_df = attack_df.select_dtypes(include=["number"])

# =========================================================
# HANDLE MISSING VALUES
# =========================================================

normal_df = normal_df.fillna(0)
attack_df = attack_df.fillna(0)

# =========================================================
# SAVE FEATURE COLUMNS
# =========================================================

columns = normal_df.columns.tolist()

# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

train_df, val_df = train_test_split(
    normal_df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("\nTrain Shape:", train_df.shape)
print("Validation Shape:", val_df.shape)

# =========================================================
# SCALING
# =========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(train_df)
X_val = scaler.transform(val_df)
X_attack = scaler.transform(attack_df)

# =========================================================
# AUTOENCODER MODEL
# =========================================================

input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim,))

encoded = Dense(64, activation="relu")(input_layer)
encoded = Dense(32, activation="relu")(encoded)

decoded = Dense(64, activation="relu")(encoded)
decoded = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(
    inputs=input_layer,
    outputs=decoded
)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.summary()

# =========================================================
# TRAIN AUTOENCODER
# =========================================================

history = autoencoder.fit(
    X_train,
    X_train,
    epochs=30,
    batch_size=256,
    validation_data=(X_val, X_val),
    verbose=1
)

# =========================================================
# PREDICTIONS
# =========================================================

X_val_pred = autoencoder.predict(X_val)
X_attack_pred = autoencoder.predict(X_attack)

# =========================================================
# RECONSTRUCTION ERROR
# =========================================================

reconstruction_error = np.mean(
    np.square(X_val - X_val_pred),
    axis=1
)

attack_error = np.mean(
    np.square(X_attack - X_attack_pred),
    axis=1
)

print("\nValidation Error Mean:", reconstruction_error.mean())
print("Attack Error Mean:", attack_error.mean())

# =========================================================
# DYNAMIC THRESHOLD
# =========================================================

window_size = 5000
percentile = 98

dynamic_thresholds = []

initial_threshold = np.percentile(
    reconstruction_error,
    percentile
)

for i in range(len(reconstruction_error)):

    if i < window_size:

        dynamic_thresholds.append(initial_threshold)

    else:

        window = reconstruction_error[i-window_size:i]

        t = np.percentile(window, percentile)

        dynamic_thresholds.append(t)

dynamic_thresholds = np.array(dynamic_thresholds)

final_threshold = float(dynamic_thresholds[-1])

print("\nFinal Threshold:", final_threshold)

# =========================================================
# AUTOENCODER PREDICTIONS
# =========================================================

y_pred_normal = (
    reconstruction_error > dynamic_thresholds
).astype(int)

y_pred_attack = (
    attack_error > final_threshold
).astype(int)

# =========================================================
# GROUND TRUTH
# =========================================================

y_true = np.concatenate([
    np.zeros(len(reconstruction_error)),
    np.ones(len(attack_error))
])

y_pred = np.concatenate([
    y_pred_normal,
    y_pred_attack
])

scores = np.concatenate([
    reconstruction_error,
    attack_error
])

# =========================================================
# AUTOENCODER METRICS
# =========================================================

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, scores)
cm = confusion_matrix(y_true, y_pred)

print("\n" + "="*60)
print("AUTOENCODER PERFORMANCE")
print("="*60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

# =========================================================
# FEATURE ERROR ANALYSIS
# =========================================================

val_feature_error = np.square(X_val - X_val_pred)
attack_feature_error = np.square(X_attack - X_attack_pred)

val_feature_mean = val_feature_error.mean(axis=0)
attack_feature_mean = attack_feature_error.mean(axis=0)

feature_importance_df = pd.DataFrame({

    "feature": columns,

    "normal_error": val_feature_mean,

    "attack_error": attack_feature_mean,

    "difference": (
        attack_feature_mean - val_feature_mean
    )

}).sort_values(
    by="difference",
    ascending=False
)

print("\nTop 10 Anomalous Features:")
print(feature_importance_df.head(10))

# =========================================================
# XGBOOST HYBRID MODEL
# =========================================================

print("\n" + "="*60)
print("XGBOOST HYBRID CLASSIFIER")
print("="*60)

X_all = np.vstack([
    X_val,
    X_attack
])

y_all = np.concatenate([
    np.zeros(len(X_val)),
    np.ones(len(X_attack))
])

ae_errors_all = np.concatenate([
    reconstruction_error,
    attack_error
]).reshape(-1, 1)

X_combined = np.hstack([
    X_all,
    ae_errors_all
])

feature_names_combined = (
    columns + ["reconstruction_error"]
)

X_xgb_train, X_xgb_test, y_xgb_train, y_xgb_test = train_test_split(
    X_combined,
    y_all,
    test_size=0.2,
    random_state=42,
    stratify=y_all
)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_xgb_train,
    y_xgb_train,
    eval_set=[
        (X_xgb_test, y_xgb_test)
    ],
    verbose=50
)

# =========================================================
# XGBOOST METRICS
# =========================================================

y_xgb_pred = xgb_model.predict(X_xgb_test)
y_xgb_prob = xgb_model.predict_proba(X_xgb_test)[:, 1]

xgb_precision = precision_score(y_xgb_test, y_xgb_pred)
xgb_recall = recall_score(y_xgb_test, y_xgb_pred)
xgb_f1 = f1_score(y_xgb_test, y_xgb_pred)
xgb_accuracy = accuracy_score(y_xgb_test, y_xgb_pred)
xgb_roc_auc = roc_auc_score(y_xgb_test, y_xgb_prob)

print("\n" + "="*60)
print("XGBOOST PERFORMANCE")
print("="*60)

print(f"Accuracy : {xgb_accuracy:.4f}")
print(f"Precision: {xgb_precision:.4f}")
print(f"Recall   : {xgb_recall:.4f}")
print(f"F1 Score : {xgb_f1:.4f}")
print(f"ROC-AUC  : {xgb_roc_auc:.4f}")

# =========================================================
# SHAP EXPLAINABILITY
# =========================================================

print("\n" + "="*60)
print("SHAP EXPLAINABILITY")
print("="*60)

background_size = min(
    500,
    X_xgb_train.shape[0]
)

X_background = X_xgb_train[:background_size]

explainer = shap.TreeExplainer(
    xgb_model,
    X_background
)

print("\nSHAP Explainer Created")

# =========================================================
# SAVE ALL ARTIFACTS
# =========================================================

print("\nSaving Artifacts...")

autoencoder.save("model.keras")

joblib.dump(
    scaler,
    "scaler.pkl"
)

joblib.dump(
    columns,
    "columns.pkl"
)

joblib.dump(
    xgb_model,
    "xgb_model.pkl"
)

joblib.dump(
    feature_names_combined,
    "xgb_feature_names.pkl"
)

joblib.dump({

    "final_threshold": final_threshold,

    "window_size": window_size,

    "percentile": percentile,

    "feature_count": len(columns),

    "xgb_enabled": True,

    "shap_enabled": True

}, "inference_config.pkl")

print("\nAll Artifacts Saved Successfully")

# =========================================================
# VISUALIZATION
# =========================================================

plt.figure(figsize=(10, 5))

plt.hist(
    reconstruction_error,
    bins=100,
    alpha=0.6,
    label="Normal"
)

plt.hist(
    attack_error,
    bins=100,
    alpha=0.6,
    label="Attack"
)

plt.axvline(
    final_threshold,
    linestyle="--",
    label="Threshold"
)

plt.legend()
plt.title("AE Reconstruction Error Distribution")
plt.show()

plt.figure()

plt.plot(
    history.history["loss"],
    label="Train Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.legend()
plt.title("Autoencoder Training Curve")
plt.show()

print("\nTraining Pipeline Completed Successfully")