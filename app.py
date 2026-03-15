# ─────────────────────────────────────────────────────────────────────────────
# app.py  —  Marketing Campaign Response Predictor (Streamlit App)
# Run with:  streamlit run app.py
# ─────────────────────────────────────────────────────────────────────────────

import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Campaign Response Predictor",
    page_icon="📊",
    layout="wide"
)

# ── Load model and threshold ──────────────────────────────────────────────────
@st.cache_resource   # loads once, stays in memory — fast reruns
def load_model():
    with open("model.pkl",     "rb") as f: model     = pickle.load(f)
    with open("threshold.pkl", "rb") as f: threshold = pickle.load(f)
    return model, threshold

model, best_threshold = load_model()

# ── Feature engineering (same logic as training) ──────────────────────────────
def build_features(inp):
    """Take raw input dict → return DataFrame with all engineered features."""
    data = pd.DataFrame([inp])

    data["TotalSpend"] = (data["MntWines"] + data["MntFruits"] + data["MntMeatProducts"]
                          + data["MntFishProducts"] + data["MntSweetProducts"] + data["MntGoldProds"])

    data["TotalChildren"]          = data["Kidhome"] + data["Teenhome"]
    data["IsParent"]               = (data["TotalChildren"] > 0).astype(int)
    data["Income_Per_Person"]      = data["Income"] / (data["TotalChildren"] + 1)
    data["TotalPurchases"]         = (data["NumWebPurchases"] + data["NumCatalogPurchases"]
                                      + data["NumStorePurchases"])
    data["AvgPurchaseValue"]       = data["TotalSpend"] / (data["TotalPurchases"] + 1)
    data["Spend_to_Income_Ratio"]  = data["TotalSpend"] / (data["Income"] + 1)
    data["TotalCampaignsAccepted"] = (data["AcceptedCmp1"] + data["AcceptedCmp2"]
                                      + data["AcceptedCmp3"] + data["AcceptedCmp4"]
                                      + data["AcceptedCmp5"])
    return data

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("📊 Marketing Campaign Response Predictor")
st.markdown(
    "Fill in the customer details below and click **Predict** "
    "to find out if this customer is likely to respond to your marketing campaign."
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# INPUT FORM  (3-column layout for a clean look)
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Personal Details**")
    year_birth     = st.number_input("Year of Birth",         min_value=1940, max_value=2005, value=1980)
    education      = st.selectbox("Education",                ["Graduation", "PhD", "Master", "2n Cycle", "Basic"])
    marital_status = st.selectbox("Marital Status",           ["Married", "Single", "Together", "Divorced", "Widow"])
    income         = st.number_input("Annual Income (₹ / $)", min_value=0,    max_value=500000, value=60000, step=1000)
    kidhome        = st.number_input("Kids at Home",          min_value=0,    max_value=5,      value=0)
    teenhome       = st.number_input("Teenagers at Home",     min_value=0,    max_value=5,      value=0)
    tenure_days    = st.number_input("Days Since Enrolled",   min_value=0,    max_value=5000,   value=365)
    recency        = st.number_input("Days Since Last Purchase (Recency)", min_value=0, max_value=365, value=30)
    complain       = st.selectbox("Filed a Complaint?",       ["No", "Yes"])

with col2:
    st.markdown("**Spending (Amount Spent in last 2 years)**")
    mnt_wines   = st.number_input("Wines (₹ / $)",           min_value=0, max_value=5000, value=200, step=10)
    mnt_fruits  = st.number_input("Fruits (₹ / $)",          min_value=0, max_value=2000, value=30,  step=5)
    mnt_meat    = st.number_input("Meat Products (₹ / $)",   min_value=0, max_value=2000, value=100, step=10)
    mnt_fish    = st.number_input("Fish Products (₹ / $)",   min_value=0, max_value=2000, value=40,  step=5)
    mnt_sweets  = st.number_input("Sweet Products (₹ / $)",  min_value=0, max_value=2000, value=30,  step=5)
    mnt_gold    = st.number_input("Gold Products (₹ / $)",   min_value=0, max_value=2000, value=50,  step=5)

    total_spend = mnt_wines + mnt_fruits + mnt_meat + mnt_fish + mnt_sweets + mnt_gold
    st.metric("💰 Total Spend (auto)", f"₹ {total_spend:,}")

with col3:
    st.markdown("**Purchase Channels**")
    num_deals   = st.number_input("Discount Purchases",   min_value=0, max_value=30, value=2)
    num_web     = st.number_input("Web Purchases",        min_value=0, max_value=30, value=3)
    num_catalog = st.number_input("Catalog Purchases",    min_value=0, max_value=30, value=2)
    num_store   = st.number_input("Store Purchases",      min_value=0, max_value=30, value=4)
    num_web_vis = st.number_input("Web Visits / Month",   min_value=0, max_value=30, value=5)

    st.markdown("**Past Campaign Responses** (1 = Accepted)")
    cmp1 = st.selectbox("Campaign 1", [0, 1], index=0)
    cmp2 = st.selectbox("Campaign 2", [0, 1], index=0)
    cmp3 = st.selectbox("Campaign 3", [0, 1], index=0)
    cmp4 = st.selectbox("Campaign 4", [0, 1], index=0)
    cmp5 = st.selectbox("Campaign 5", [0, 1], index=0)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT BUTTON
# ══════════════════════════════════════════════════════════════════════════════
predict_clicked = st.button("🔍 Predict Campaign Response", use_container_width=True)

if predict_clicked:
    # Assemble raw input dictionary
    customer_input = {
        "Year_Birth"          : year_birth,
        "Education"           : education,
        "Marital_Status"      : marital_status,
        "Income"              : income,
        "Kidhome"             : kidhome,
        "Teenhome"            : teenhome,
        "Recency"             : recency,
        "MntWines"            : mnt_wines,
        "MntFruits"           : mnt_fruits,
        "MntMeatProducts"     : mnt_meat,
        "MntFishProducts"     : mnt_fish,
        "MntSweetProducts"    : mnt_sweets,
        "MntGoldProds"        : mnt_gold,
        "NumDealsPurchases"   : num_deals,
        "NumWebPurchases"     : num_web,
        "NumCatalogPurchases" : num_catalog,
        "NumStorePurchases"   : num_store,
        "NumWebVisitsMonth"   : num_web_vis,
        "AcceptedCmp1"        : cmp1,
        "AcceptedCmp2"        : cmp2,
        "AcceptedCmp3"        : cmp3,
        "AcceptedCmp4"        : cmp4,
        "AcceptedCmp5"        : cmp5,
        "Complain"            : 1 if complain == "Yes" else 0,
        "Tenure_Days"         : tenure_days,
    }

    # Build features and predict
    data        = build_features(customer_input)
    probability = model.predict_proba(data)[0][1]
    prediction  = 1 if probability >= best_threshold else 0

    # ── Result display ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.metric("Respond Probability", f"{probability * 100:.1f}%")

    with res_col2:
        st.metric("Decision Threshold", f"{best_threshold:.2f}")

    with res_col3:
        st.metric("Total Campaigns Accepted", int(cmp1 + cmp2 + cmp3 + cmp4 + cmp5))

    # Big result banner
    if prediction == 1:
        st.success("✅  This customer is likely to RESPOND to the campaign!")
    else:
        st.error("❌  This customer is likely NOT to respond to the campaign.")

    # Recommendation message
    st.markdown("### 💡 Recommendation")
    if probability >= 0.6:
        st.info("🔥 **HIGH chance** — Definitely include this customer. Strong ROI expected.")
    elif probability >= best_threshold:
        st.info("✅ **MODERATE chance** — Worth including in the campaign.")
    elif probability >= 0.25:
        st.warning("⚠️ **LOW chance** — Consider cheaper outreach (email only, no call).")
    else:
        st.warning("🚫 **VERY LOW chance** — Skip this customer. Not cost-effective to target.")

    # Customer summary card
    st.markdown("### 📋 Customer Summary")
    summary_col1, summary_col2 = st.columns(2)

    age = 2025 - year_birth
    total_kids = kidhome + teenhome

    with summary_col1:
        st.markdown(f"""
| Field | Value |
|---|---|
| Age | {age} years |
| Education | {education} |
| Marital Status | {marital_status} |
| Children at Home | {total_kids} |
| Annual Income | ₹ {income:,} |
| Days Since Last Purchase | {recency} days |
        """)

    with summary_col2:
        st.markdown(f"""
| Field | Value |
|---|---|
| Total Spend | ₹ {total_spend:,} |
| Total Purchases | {num_web + num_catalog + num_store} |
| Past Campaigns Accepted | {cmp1 + cmp2 + cmp3 + cmp4 + cmp5} |
| Web Visits / Month | {num_web_vis} |
| Filed Complaint | {complain} |
| Member For | {tenure_days} days |
        """)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<center><small>Marketing Campaign Response Predictor · "
    "Random Forest + SMOTE · Built with Streamlit</small></center>",
    unsafe_allow_html=True
)
