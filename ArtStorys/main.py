import streamlit as st

# Titel der App
st.title("Ranking von Art Stories")

# Untertitel
st.subheader("Finden Sie die ähnlichsten Geschichten zu Ihrer Anfrage")

# Auswahl des Embeddings
embedding_type = st.selectbox(
    "Wählen Sie das Embedding aus:", ["Word2Vec", "Doc2Vec"]
)

# Eingabefeld für die Nutzeranfrage
user_query = st.text_input("Geben Sie Ihre Anfrage ein:", placeholder="Beschreiben Sie Ihre Suche hier...")

# Button zur Bestätigung der Eingabe
if st.button("Ähnliche Geschichten finden"):
    if user_query.strip():
        st.write(f"Ihre Anfrage: '{user_query}' wird verarbeitet mit {embedding_type}...")
        # Hier könnte die Ähnlichkeitsberechnung eingebunden werden
    else:
        st.warning("Bitte geben Sie eine gültige Anfrage ein.")
