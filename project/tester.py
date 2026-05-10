import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, roc_auc_score
)
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt
import joblib

# ── UPGRADE 1 & 2 imports ──────────────────────────────────────────────────
from xgboost import XGBClassifier
import shap
# ──────────────────────────────────────────────────────────────────────────

np.random.seed(42)
tf.random.set_seed(42)

# 📂 Load dataset
df = pd.read_csv(
    r"C:\Users\TEJAS\OneDrive\Desktop\PROJECTS\IDS\project\data\UNSW-NB15_1_with_features.csv",
    low_memory=False
)

print("Original shape:", df.shape)

# 🧹 Clean label
df["Label"] = pd.to_numeric(df["Label"], errors="coerce")

print("\nLabel distribution:")
print(df["Label"].value_counts())

# Split
normal_df = df[df["Label"] == 0]
attack_df = df[df["Label"] == 1]

print("\nNormal shape:", normal_df.shape)
print("Attack shape:", attack_df.shape)

# ❌ Drop label
normal_df = normal_df.drop(columns=["Label"])
attack_df = attack_df.drop(columns=["Label"])

# 🔢 Keep numeric
normal_df = normal_df.select_dtypes(include=["number"])
attack_df = attack_df.select_dtypes(include=["number"])

# 🛠 Handle missing
normal_df = normal_df.fillna(0)
attack_df = attack_df.fillna(0)

# 🔒 Save columns
columns = normal_df.columns.tolist()

# Split train/val
train_df, val_df = train_test_split(
    normal_df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

val_df = val_df.fillna(0)

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)

# ⚙️ Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(train_df)
X_val = scaler.transform(val_df)
X_attack = scaler.transform(attack_df)

# 🧠 Autoencoder
input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim,))

encoded = Dense(64, activation="relu")(input_layer)
encoded = Dense(32, activation="relu")(encoded)

decoded = Dense(64, activation="relu")(encoded)
decoded = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.summary()

# ⏱ Training
history = autoencoder.fit(
    X_train,
    X_train,
    epochs=30,
    batch_size=256,
    validation_data=(X_val, X_val),
    verbose=1
)

# 🔍 Predictions
X_val_pred = autoencoder.predict(X_val)
X_attack_pred = autoencoder.predict(X_attack)

# 📉 Reconstruction errors
reconstruction_error = np.mean(np.square(X_val - X_val_pred), axis=1)
attack_error = np.mean(np.square(X_attack - X_attack_pred), axis=1)

print("\nValidation Error Mean:", reconstruction_error.mean())
print("Attack Error Mean:", attack_error.mean())

# =========================================================
# 🚀 DYNAMIC THRESHOLD (MOVING WINDOW)
# =========================================================

window_size = 5000
percentile = 98

dynamic_thresholds = []

initial_threshold = np.percentile(reconstruction_error, percentile)

for i in range(len(reconstruction_error)):
    if i < window_size:
        dynamic_thresholds.append(initial_threshold)
    else:
        window = reconstruction_error[i-window_size:i]
        t = np.percentile(window, percentile)
        dynamic_thresholds.append(t)

dynamic_thresholds = np.array(dynamic_thresholds)

# Final threshold (latest)
final_threshold = dynamic_thresholds[-1]

# =========================================================
# 🚨 PREDICTIONS
# =========================================================

y_pred_normal = (reconstruction_error > dynamic_thresholds).astype(int)
y_pred_attack = (attack_error > final_threshold).astype(int)

# Ground truth
y_true = np.concatenate([
    np.zeros(len(reconstruction_error)),
    np.ones(len(attack_error))
])

y_pred = np.concatenate([y_pred_normal, y_pred_attack])


# =========================================================
# 🔍 FEATURE-LEVEL ERROR ANALYSIS
# =========================================================

# Per-feature reconstruction error
val_feature_error = np.square(X_val - X_val_pred)
attack_feature_error = np.square(X_attack - X_attack_pred)

# Mean error per feature
val_feature_mean = val_feature_error.mean(axis=0)
attack_feature_mean = attack_feature_error.mean(axis=0)

feature_importance_df = pd.DataFrame({
    "feature": columns,
    "normal_error": val_feature_mean,
    "attack_error": attack_feature_mean,
    "difference": attack_feature_mean - val_feature_mean
}).sort_values(by="difference", ascending=False)

print("\n🔥 Top Features Causing Anomalies:")
print(feature_importance_df.head(10))

# =========================================================
# 📊 AUTOENCODER METRICS
# =========================================================

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\n📊 AUTOENCODER MODEL PERFORMANCE")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()

print(f"\nTP: {tp}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TN: {tn}")

# =========================================================
# 🚨 DETECTION RATE (IMPORTANT FOR IDS)
# =========================================================

detection_rate = (tp / (tp + fn)) * 100

print(f"\n Detection Rate: {detection_rate:.2f}%")

# =========================================================
# 📈 ROC-AUC
# =========================================================

scores = np.concatenate([reconstruction_error, attack_error])
roc_auc = roc_auc_score(y_true, scores)

print(f"\nROC-AUC Score: {roc_auc:.4f}")


autoencoder.save("model.keras")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(columns, "columns.pkl")

joblib.dump({
    "window_size": window_size,
    "percentile": percentile,
    "initial_threshold": float(initial_threshold)
}, "threshold_config.pkl")

print("\n✅ Autoencoder artifacts saved successfully!")


# =========================================================
# 🚀 UPGRADE 1 — XGBoost HYBRID CLASSIFIER
# =========================================================
# Your AE flags anomalies. XGBoost then classifies them precisely.
# The reconstruction error itself is added as an extra feature — it
# is one of the strongest signals for separating normal from attack.

print("\n" + "="*60)
print("🚀 UPGRADE 1: XGBoost Hybrid Classifier")
print("="*60)

# Stack val (normal) and attack data back together with labels
X_all = np.vstack([X_val, X_attack])
y_all = np.concatenate([
    np.zeros(len(X_val)),
    np.ones(len(X_attack))
])

# Append reconstruction error as an additional feature
ae_errors_all = np.concatenate([reconstruction_error, attack_error]).reshape(-1, 1)
X_combined = np.hstack([X_all, ae_errors_all])

# Feature names for SHAP (original columns + the new error feature)
feature_names_combined = columns + ["reconstruction_error"]

# Train / test split for XGBoost
X_xgb_train, X_xgb_test, y_xgb_train, y_xgb_test = train_test_split(
    X_combined, y_all,
    test_size=0.2,
    random_state=42,
    stratify=y_all
)

print(f"\nXGBoost train size : {X_xgb_train.shape[0]}")
print(f"XGBoost test  size : {X_xgb_test.shape[0]}")

# Train XGBoost
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_xgb_train, y_xgb_train,
    eval_set=[(X_xgb_test, y_xgb_test)],
    verbose=50
)

# XGBoost predictions
y_xgb_pred = xgb_model.predict(X_xgb_test)
y_xgb_prob = xgb_model.predict_proba(X_xgb_test)[:, 1]

# XGBoost metrics
xgb_precision = precision_score(y_xgb_test, y_xgb_pred)
xgb_recall    = recall_score(y_xgb_test, y_xgb_pred)
xgb_f1        = f1_score(y_xgb_test, y_xgb_pred)
xgb_accuracy  = accuracy_score(y_xgb_test, y_xgb_pred)
xgb_roc_auc   = roc_auc_score(y_xgb_test, y_xgb_prob)
xgb_cm        = confusion_matrix(y_xgb_test, y_xgb_pred)

print("\n📊 XGBOOST MODEL PERFORMANCE")
print(f"Accuracy : {xgb_accuracy:.4f}")
print(f"Precision: {xgb_precision:.4f}")
print(f"Recall   : {xgb_recall:.4f}")
print(f"F1 Score : {xgb_f1:.4f}")
print(f"ROC-AUC  : {xgb_roc_auc:.4f}")

print("\nConfusion Matrix:")
print(xgb_cm)

xgb_tn, xgb_fp, xgb_fn, xgb_tp = xgb_cm.ravel()
xgb_detection_rate = (xgb_tp / (xgb_tp + xgb_fn)) * 100

print(f"\nTP: {xgb_tp}  FP: {xgb_fp}  FN: {xgb_fn}  TN: {xgb_tn}")
print(f"Detection Rate (XGBoost): {xgb_detection_rate:.2f}%")

# ── Comparison summary ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("📊 AE vs AE + XGBoost COMPARISON")
print("="*60)
print(f"{'Metric':<20} {'Autoencoder':>15} {'AE + XGBoost':>15}")
print("-"*50)
print(f"{'Accuracy':<20} {accuracy:>15.4f} {xgb_accuracy:>15.4f}")
print(f"{'Precision':<20} {precision:>15.4f} {xgb_precision:>15.4f}")
print(f"{'Recall':<20} {recall:>15.4f} {xgb_recall:>15.4f}")
print(f"{'F1 Score':<20} {f1:>15.4f} {xgb_f1:>15.4f}")
print(f"{'ROC-AUC':<20} {roc_auc:>15.4f} {xgb_roc_auc:>15.4f}")
print(f"{'Detection Rate':<20} {detection_rate:>14.2f}% {xgb_detection_rate:>14.2f}%")

# Save XGBoost model
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(feature_names_combined, "xgb_feature_names.pkl")
print("\n✅ XGBoost model saved as xgb_model.pkl")


# =========================================================
# 🔍 UPGRADE 2 — SHAP EXPLAINABILITY
# =========================================================
# For every alert the model raises, SHAP tells you WHICH features
# drove that decision. Essential for real SOC/analyst workflows.

print("\n" + "="*60)
print("🔍 UPGRADE 2: SHAP Explainability")
print("="*60)

# Use a background sample for speed (500 rows is enough for TreeExplainer)
background_size = min(500, X_xgb_train.shape[0])
X_background = X_xgb_train[:background_size]

explainer = shap.TreeExplainer(xgb_model, X_background)

# Explain the test set (cap at 1000 rows for speed)
explain_size = min(1000, X_xgb_test.shape[0])
X_explain = X_xgb_test[:explain_size]

print(f"\nComputing SHAP values for {explain_size} samples...")
shap_values = explainer.shap_values(X_explain)

# ── Per-sample top features ──────────────────────────────────────────────
# For each flagged attack in the test set, print top 3 driving features
attack_indices = np.where(y_xgb_pred[:explain_size] == 1)[0]

print("\n🔎 Top 3 features driving each of the first 5 alerts:")
for i, idx in enumerate(attack_indices[:5]):
    sample_shap = shap_values[idx]
    top3_idx = np.argsort(np.abs(sample_shap))[-3:][::-1]
    print(f"\n  Alert #{i+1} (sample index {idx})")
    for rank, feat_idx in enumerate(top3_idx, 1):
        fname = feature_names_combined[feat_idx]
        fval  = X_explain[idx][feat_idx]
        sval  = sample_shap[feat_idx]
        direction = "↑ pushed toward ATTACK" if sval > 0 else "↓ pushed toward NORMAL"
        print(f"    {rank}. {fname:<30} value={fval:>10.4f}  SHAP={sval:>8.4f}  {direction}")

# ── Global feature importance from SHAP ─────────────────────────────────
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    "feature": feature_names_combined,
    "mean_abs_shap": mean_abs_shap
}).sort_values(by="mean_abs_shap", ascending=False)

print("\n🏆 Global SHAP Feature Importance (top 10):")
print(shap_importance_df.head(10).to_string(index=False))

# Save SHAP explainer
joblib.dump(explainer, "shap_explainer.pkl")
print("\n✅ SHAP explainer saved as shap_explainer.pkl")


# =========================================================
# 📊 VISUALIZATION
# =========================================================

# Original AE error distribution
plt.figure(figsize=(10, 5))
plt.hist(reconstruction_error, bins=100, alpha=0.6, label="Normal")
plt.hist(attack_error, bins=100, alpha=0.6, label="Attack")
plt.axvline(final_threshold, linestyle="--", label="Threshold")
plt.legend()
plt.title("AE Reconstruction Error Distribution")
plt.show()

# Training curve
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Autoencoder Training Curve")
plt.show()

# ── UPGRADE 1: XGBoost feature importance ───────────────────────────────
plt.figure(figsize=(10, 6))
xgb_fi = pd.Series(
    xgb_model.feature_importances_,
    index=feature_names_combined
).sort_values(ascending=False).head(15)
xgb_fi.plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("XGBoost Feature Importance (top 15)")
plt.tight_layout()
plt.show()

# ── UPGRADE 2: SHAP summary plot ────────────────────────────────────────
print("\nGenerating SHAP summary plot...")
shap.summary_plot(
    shap_values,
    X_explain,
    feature_names=feature_names_combined,
    max_display=15,
    show=True
)

# SHAP bar plot (mean absolute SHAP — clean for reports/presentations)
shap.summary_plot(
    shap_values,
    X_explain,
    feature_names=feature_names_combined,
    plot_type="bar",
    max_display=15,
    show=True
)

print("\n✅ All artifacts saved:")
print("   model.keras          — autoencoder")
print("   scaler.pkl           — StandardScaler")
print("   columns.pkl          — feature column list")
print("   threshold_config.pkl — dynamic threshold config")
print("   xgb_model.pkl        — XGBoost hybrid classifier  [UPGRADE 1]")
print("   xgb_feature_names.pkl— feature names for XGBoost  [UPGRADE 1]")
print("   shap_explainer.pkl   — SHAP TreeExplainer          [UPGRADE 2]")