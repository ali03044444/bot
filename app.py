
import streamlit as st
from unified_bot import train_model, predict_and_trade, backtest

st.set_page_config(page_title="OKX LSTM Trading Bot", layout="centered")

st.title("🚀 OKX LSTM Auto-Trading Bot")
st.markdown("Train, Predict & Trade from one interface. Make sure you have API keys in the code!")

menu = st.sidebar.radio("Choose Action", ["📊 Train Model", "📈 Predict & Trade", "🔁 Backtest"])

if menu == "📊 Train Model":
    if st.button("🚀 Start Training"):
        with st.spinner("Training model..."):
            train_model()
        st.success("✅ Model trained and saved!")

elif menu == "📈 Predict & Trade":
    if st.button("📡 Run Prediction & Execute"):
        with st.spinner("Predicting and executing..."):
            predict_and_trade()
        st.success("✅ Prediction done. Check terminal for trades.")

elif menu == "🔁 Backtest":
    if st.button("🔍 Run Backtest"):
        with st.spinner("Running backtest..."):
            backtest()
        st.success("✅ Backtest complete.")
