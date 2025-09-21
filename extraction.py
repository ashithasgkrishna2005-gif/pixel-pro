"{PRESCRIPTION_TEXT}"
# backend/extraction.py
def extract_with_llm(gen_pipeline, text):
    prompt = f"""
    Extract structured prescription items from the text below. Output strictly valid JSON with keys:
    patient_age, medications (array of objects: name, strength, dose, frequency, route, duration, raw_text).
    Text:
    \"\"\"{text}\"\"\"
    """
    out = gen_pipeline(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]
    # the model might return some header — try to find the JSON substring
    import json, re
    m = re.search(r"\{.*\}", out, re.S)
    if m:
        return json.loads(m.group(0))
    else:
        raise ValueError("Could not parse JSON from model output: " + out[:300])
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
        from backend.extraction import extract_with_llm
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
