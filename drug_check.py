# backend/drug_check.py
import pandas as pd

interactions = pd.read_csv("data/interactions_sample.csv")

def check_interactions(detected_drugs):
    results = []
    for i, d1 in enumerate(detected_drugs):
        for d2 in detected_drugs[i+1:]:
            row = interactions[(interactions.drug_a.str.lower()==d1.lower()) & (interactions.drug_b.str.lower()==d2.lower()) |
                               (interactions.drug_a.str.lower()==d2.lower()) & (interactions.drug_b.str.lower()==d1.lower())]
            if not row.empty:
                results.append(row.iloc[0].to_dict())
    return results
# app.py (Streamlit prototype)
import streamlit as st
import os, json
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Prescription Verifier", layout="wide")

st.title("AI Medical Prescription Verifier — Prototype (Demo Only)")
st.markdown("**Warning:** Demo only. Not for clinical use. Always verify with a clinician.")

prescription_text = st.text_area("Paste prescription / notes here", height=200)
patient_age = st.number_input("Patient age (years)", min_value=0, max_value=130, value=30)
analyze = st.button("Analyze prescription")

if analyze:
    with st.spinner("Loading models and analyzing..."):
        # Lazy load to keep startup fast for testing
        from backend.drug_check import check_interactions
        tokenizer, model, gen = load_models(device_map="auto")

        try:
            parsed = extract_with_llm(gen, prescription_text)
        except Exception as e:
            st.error("Extraction failed: " + str(e))
            st.stop()

        st.subheader("Extracted data")
        st.json(parsed)

        # Check interactions
        meds = [m.get("name") for m in parsed.get("medications",[])]
        inter = check_interactions(meds)
        if inter:
            st.error(f"⚠️ Found {len(inter)} interaction(s):")
            for r in inter:
                st.write(f"- {r['drug_a']} ↔ {r['drug_b']}: {r['severity']} — {r.get('notes')}")
        else:
            st.success("No known interactions found in local dataset (demo).")

        # Age disclaimer (demo)
        st.info("Age-specific dosing: demo only. Replace with validated dosing tables before any clinical use.")
