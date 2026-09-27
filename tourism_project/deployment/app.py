import json

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

MODEL_REPO = "shrimathin/tourism-wellness-model"

st.set_page_config(page_title="Wellness Tourism Purchase Predictor", page_icon="🧘", layout="wide")


@st.cache_resource(show_spinner="Loading the latest model from Hugging Face…")
def load_model():
    model = joblib.load(hf_hub_download(MODEL_REPO, "wellness_model.joblib"))
    meta = json.load(open(hf_hub_download(MODEL_REPO, "model_metadata.json")))
    return model, meta


model, meta = load_model()
THRESHOLD = meta["threshold"]
FEATURES = meta["features"]

with st.sidebar:
    st.header("Model card")
    st.write(f"**Algorithm:** {meta['model_name']}")
    st.write(f"**Trained:** {meta['trained_at']} (commit `{meta['git_sha']}`)")
    st.write(f"**Decision threshold:** {THRESHOLD:.2f}")
    m = meta["test_metrics"]
    st.metric("Test recall", f"{m['recall']:.1%}")
    st.metric("Test precision", f"{m['precision']:.1%}")
    st.metric("Test ROC-AUC", f"{m['roc_auc']:.3f}")
    st.caption(f"Model repo: huggingface.co/{MODEL_REPO}")

st.title("🧘 Wellness Tourism Package: Purchase Predictor")
st.write("Score customers **before** the sales call so the team can focus on the people most likely to buy.")

single, batch = st.tabs(["Single customer", "Batch scoring (CSV)"])

with single:
    with st.form("customer"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Profile")
            age = st.slider("Age", 18, 70, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital = st.selectbox("Marital status", ["Single", "Unmarried", "Married", "Divorced"])
            occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
            income = st.number_input("Monthly income", 1000, 150000, 22000, step=500)
            city_tier = st.selectbox("City tier", [1, 2, 3])
        with c2:
            st.subheader("Travel")
            passport = st.radio("Has passport?", ["Yes", "No"], horizontal=True)
            own_car = st.radio("Owns a car?", ["Yes", "No"], horizontal=True)
            trips = st.number_input("Trips per year", 0, 25, 3)
            persons = st.number_input("People visiting", 1, 10, 3)
            children = st.number_input("Children (<5) visiting", 0, 5, 1)
            star = st.selectbox("Preferred property star", [3, 4, 5])
        with c3:
            st.subheader("Sales interaction")
            contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
            product = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
            duration = st.slider("Pitch duration (min)", 5, 60, 15)
            followups = st.slider("Number of follow-ups", 1, 6, 4)
            satisfaction = st.slider("Pitch satisfaction score", 1, 5, 3)
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "Age": age, "CityTier": city_tier, "DurationOfPitch": duration, "NumberOfPersonVisiting": persons,
            "NumberOfFollowups": followups, "PreferredPropertyStar": star, "NumberOfTrips": trips,
            "Passport": int(passport == "Yes"), "PitchSatisfactionScore": satisfaction,
            "OwnCar": int(own_car == "Yes"), "NumberOfChildrenVisiting": children, "MonthlyIncome": income,
            "TypeofContact": contact, "Occupation": occupation, "Gender": gender,
            "ProductPitched": product, "MaritalStatus": marital,
        }])[FEATURES]
        p = float(model.predict_proba(row)[0, 1])
        k1, k2 = st.columns([1, 2])
        k1.metric("Purchase probability", f"{p:.1%}")
        k1.progress(min(p, 1.0))
        if p >= THRESHOLD:
            k2.success("✅ **High-propensity customer.** Prioritise for a Wellness package call.")
        else:
            k2.warning("⏸️ **Low propensity.** Keep in nurture campaigns rather than a direct call.")
        tips = []
        if passport == "No": tips.append("No passport: domestic wellness options may resonate more.")
        if followups < 4: tips.append("More follow-ups are strongly associated with conversion.")
        if product in ("Super Deluxe", "King"): tips.append("Premium pitches convert rarely. Consider leading with Basic/Standard.")
        if tips:
            k2.info("**Suggestions:**\n\n- " + "\n- ".join(tips))

with batch:
    st.write(f"Upload a CSV containing these columns: `{', '.join(FEATURES)}`")
    up = st.file_uploader("Customer file", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        if "Gender" in df:
            df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            df["purchase_probability"] = model.predict_proba(df[FEATURES])[:, 1]
            df["call_recommended"] = df["purchase_probability"] >= THRESHOLD
            df = df.sort_values("purchase_probability", ascending=False)
            st.write(f"**{int(df['call_recommended'].sum())} of {len(df)}** customers recommended for a call.")
            st.dataframe(df, use_container_width=True)
            st.download_button("Download ranked call list", df.to_csv(index=False), "call_list.csv", "text/csv")
