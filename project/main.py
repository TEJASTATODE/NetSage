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
# 📊 METRICS
# =========================================================


precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\n📊 MODEL PERFORMANCE")
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

print("\n✅ All artifacts saved successfully!")

# =========================================================
# 📊 VISUALIZATION
# =========================================================

plt.figure(figsize=(10, 5))

plt.hist(reconstruction_error, bins=100, alpha=0.6, label="Normal")
plt.hist(attack_error, bins=100, alpha=0.6, label="Attack")

plt.axvline(final_threshold, linestyle="--", label="Threshold")

plt.legend()
plt.title("Error Distribution")
plt.show()

# Training curve
plt.figure()

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")

plt.legend()
plt.title("Training Curve")
plt.show()