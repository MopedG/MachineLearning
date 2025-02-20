import streamlit as st
import main

st.title("Chain-of-Thought Prompting")

st.subheader("Modellauswahl")
st.radio(options=["Gemini 1.5 Pro (>200B)", "llama3.1"])

st.subheader("Funktion")
option = st.radio(options=["freies Prompting", "Benchmark"])

st.divider()

if option == "freies Prompting":
    st.subheader("Chain-of-Thought-Auswahl")
    st.radio(options=["Zero-Shot", "Few-Shot"])
    st.checkbox("Vergleiche mit Antwort, die kein Chain-of-Thought verwendet")

st.divider()

main.is_gemini_available()
















