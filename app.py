import streamlit as st
from bert_model import predict_sentiment

st.set_page_config(page_title="Sentiment Analysis with BERT", layout="centered")

st.markdown("## 📊 Sentiment Analysis with BERT")
st.markdown("Enter a product review below:")

user_input = st.text_area("📝 Your Review:", height=150)

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip():
        prediction = predict_sentiment(user_input)
        st.success(f"✅ Sentiment: **{prediction}**")
    else:
        st.warning("Please enter some text to analyze.")
