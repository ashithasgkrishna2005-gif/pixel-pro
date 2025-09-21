from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access them in your code
hf_token = os.getenv("HUGGINGFACE_TOKEN")
ibm_key = os.getenv("IBM_API_KEY")
ibm_url = os.getenv("IBM_URL")

print("HF Token:", hf_token[:10] + "...")   # Just to check (don’t print full secrets!)
# backend/model_loader.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

HF_BASE = "ibm-granite/granite-guardian-3.2-5b"
LORA_ADAPTER = "ibm-granite/granite-guardian-3.2-5b-lora-harm-correction"

def load_models(device_map="auto"):
    # tokenizer & base
    tokenizer = AutoTokenizer.from_pretrained(HF_BASE, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        HF_BASE,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,  # sometimes required for custom tokenizers/chat templates
    )
    # apply LoRA adapter (PEFT)
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
    # small pipeline wrapper (you might call model.generate directly for chat templates)
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map=device_map)
    return tokenizer, model, gen

# Example: tokenizer, model, gen = load_models()
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
        from backend.model_loader import load_models
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
You are an assistant that extracts prescription information and returns strict JSON only.
Return JSON with keys: patient_age (int or null), medications (array).
For each medication provide: name, strength, dose, frequency, route, duration, raw_text.
Text:
"Amoxicillin 500 mg, 1 tablet TID for 7 days. Paracetamol 500 mg PRN q6h. Patient age: 8 yrs."
