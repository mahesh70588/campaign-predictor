# ─────────────────────────────────────────────────────────────────────────────
# train_model.py
# Run this file ONCE to train the model and save it as model.pkl + threshold.pkl
# Command: python train_model.py
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              roc_auc_score, precision_recall_curve)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

print("=" * 55)
print("  MARKETING CAMPAIGN MODEL — TRAINING SCRIPT")
print("=" * 55)

# ── Step 1: Load data ─────────────────────────────────────────
print("\n[1/7] Loading data...")
df = pd.read_csv("marketing_campaign_enhanced.csv")
print(f"      Loaded {df.shape[0]} rows, {df.shape[1]} columns.")

# ── Step 2: Clean data ────────────────────────────────────────
print("[2/7] Cleaning data...")
df["Income"] = df["Income"].fillna(df["Income"].median())
df = df.drop_duplicates()

# ── Step 3: Feature engineering ──────────────────────────────
print("[3/7] Engineering features...")
df["TotalSpend"] = (df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"]
                    + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"])

df["TotalChildren"]          = df["Kidhome"] + df["Teenhome"]
df["IsParent"]               = (df["TotalChildren"] > 0).astype(int)
df["Income_Per_Person"]      = df["Income"] / (df["TotalChildren"] + 1)
df["TotalPurchases"]         = df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]
df["AvgPurchaseValue"]       = df["TotalSpend"] / (df["TotalPurchases"] + 1)
df["Spend_to_Income_Ratio"]  = df["TotalSpend"] / (df["Income"] + 1)
df["TotalCampaignsAccepted"] = (df["AcceptedCmp1"] + df["AcceptedCmp2"] + df["AcceptedCmp3"]
                                 + df["AcceptedCmp4"] + df["AcceptedCmp5"])

# ── Step 4: Prepare X and y ───────────────────────────────────
print("[4/7] Preparing features and target...")
df = df.drop("ID", axis=1)
X  = df.drop("Response", axis=1)
y  = df["Response"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Step 5: Build preprocessor ───────────────────────────────
print("[5/7] Building preprocessing pipeline...")
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

num_pipeline = SklearnPipeline([
    ("fill_missing", SimpleImputer(strategy="median")),
    ("scale",        StandardScaler())
])
cat_pipeline = SklearnPipeline([
    ("fill_missing", SimpleImputer(strategy="most_frequent")),
    ("encode",       OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("numeric",     num_pipeline, num_cols),
    ("categorical", cat_pipeline, cat_cols)
])

# ── Step 6: Train with GridSearchCV ──────────────────────────
print("[6/7] Training model with GridSearchCV (this may take 2-5 minutes)...")

param_grid = {
    "model__n_estimators" : [100, 200],
    "model__max_depth"    : [5, 10, None],
    "model__class_weight" : ["balanced"]
}

rf_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("smote",      SMOTE(random_state=42)),
    ("model",      RandomForestClassifier(random_state=42))
])

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"      Best params: {grid_search.best_params_}")
print(f"      Best CV F1 : {grid_search.best_score_:.4f}")

# ── Step 7: Find optimal threshold ───────────────────────────
print("[7/7] Finding optimal decision threshold...")
probs = best_model.predict_proba(X_test)[:, 1]
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, probs)

best_threshold = 0.5
best_f1        = 0.0

for p, r, t in zip(precision_vals, recall_vals, thresholds):
    if r >= 0.60 and p >= 0.30:
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1        = f1
            best_threshold = t

print(f"      Optimal threshold: {best_threshold:.2f}")

# ── Evaluate ─────────────────────────────────────────────────
y_pred = (probs >= best_threshold).astype(int)
print(f"\nFinal Evaluation (threshold = {best_threshold:.2f}):")
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_test, probs):.4f}")
print(classification_report(y_test, y_pred, target_names=["Not Respond", "Respond"]))

# ── Save model and threshold ──────────────────────────────────
with open("model.pkl",     "wb") as f: pickle.dump(best_model,     f)
with open("threshold.pkl", "wb") as f: pickle.dump(best_threshold, f)

print("=" * 55)
print("  model.pkl     — saved ✅")
print("  threshold.pkl — saved ✅")
print("  Training complete! Now run:  streamlit run app.py")
print("=" * 55)
