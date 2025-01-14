import streamlit as st
import os

st.title("Ranking von Art Stories")

st.subheader("Finden Sie die ähnlichsten Geschichten zu Ihrer Anfrage")

embedding_type = st.selectbox(
    "Wählen Sie das Embedding aus:", ["Word2Vec", "Doc2Vec"]
)

user_query = st.text_input("Geben Sie Ihre Anfrage ein:", placeholder="Beschreiben Sie Ihre Suche hier...")

if st.button("Ähnliche Geschichten finden"):
    if user_query.strip():
        st.write(f"Ihre Anfrage: '{user_query}' wird verarbeitet mit {embedding_type}...")
    else:
        st.warning("Bitte geben Sie eine gültige Anfrage ein.")
