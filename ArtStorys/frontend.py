import streamlit as st

st.title("Ranking von Art Stories")
st.subheader("Finden Sie die ähnlichsten Geschichten zu Ihrer Anfrage")

# Auswahl des Embedding-Typs
embedding_type = st.selectbox(
    "Wählen Sie das Embedding aus:", ["Word2Vec", "Doc2Vec"]
)

# Kurze Beschreibung der Auswirkungen des jeweiligen Modells
if embedding_type == "Word2Vec":
    st.markdown("""
    **Word2Vec**: Dieses Modell wandelt jedes Wort in deiner Anfrage in einen Vektor um. 
    Ähnliche Geschichten werden anhand der Ähnlichkeit der Wortvektoren verglichen.
    Dies kann besonders hilfreich sein, wenn du suchst, wie ähnliche Begriffe in deinem Text verwendet werden.
    """)
elif embedding_type == "Doc2Vec":
    st.markdown("""
    **Doc2Vec**: Dieses Modell wandelt die gesamte Anfrage in einen einzigen Vektor um. 
    Ähnliche Geschichten werden basierend auf dem Gesamtzusammenhang deiner Anfrage verglichen. 
    Dies eignet sich gut für längere Texte oder Anfragen, die über einzelne Wörter hinausgehen.
    """)

# Eingabefeld für die Anfrage
user_query = st.text_input("Geben Sie Ihre Anfrage ein:", placeholder="Beschreiben Sie Ihre Suche hier...")

# Button zum Suchen der ähnlichen Geschichten
if st.button("Ähnliche Geschichten finden"):
    if user_query.strip():
        st.write(f"Ihre Anfrage: '{user_query}' wird verarbeitet mit {embedding_type}...")
    else:
        st.warning("Bitte geben Sie eine gültige Anfrage ein.")
