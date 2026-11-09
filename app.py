"""Streamlit UI: paste a message (or upload a CSV) and get Spam / Not Spam."""
import pandas as pd
import streamlit as st

from src.predict import predict_message

st.set_page_config(page_title="Spam Message Classifier", page_icon="📨")
st.title("📨 Spam Message Classifier")
st.write("Paste an SMS, email body or support ticket below.")

text = st.text_area("Message", height=160, placeholder="e.g. WINNER!! Claim your free prize now...")

if st.button("Predict", type="primary"):
    if not text.strip():
        st.warning("Please enter a message first.")
    else:
        result = predict_message(text)
        prob = result["spam_probability"]
        if result["label"] == "Spam":
            st.error(f"🚫 **Spam** - confidence {prob:.1%}")
        else:
            st.success(f"✅ **Not Spam** - confidence {1 - prob:.1%}")
        st.progress(prob, text=f"Spam probability: {prob:.1%}")

st.divider()
st.subheader("Batch check (optional)")
uploaded = st.file_uploader("Upload a CSV with a 'message' column", type="csv")
if uploaded is not None:
    df = pd.read_csv(uploaded)
    if "message" not in df.columns:
        st.error("CSV must contain a 'message' column.")
    else:
        preds = [predict_message(m) for m in df["message"].astype(str)]
        df["prediction"] = [p["label"] for p in preds]
        df["spam_probability"] = [round(p["spam_probability"], 4) for p in preds]
        st.dataframe(df, use_container_width=True)
        st.download_button("Download results", df.to_csv(index=False), "predictions.csv")
