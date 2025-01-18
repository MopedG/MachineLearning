import streamlit as st
import main
st.title("Ranking von Art Stories")
st.subheader("Finden Sie die ähnlichsten Geschichten zu Ihrer Anfrage")

# Auswahlmöglichkeit zwischen Word2Vec und Doc2Vec
isWord2Vec = st.radio(
    "Wählen Sie den Algorithmus für die Suche:",
    ("Word2Vec", "Doc2Vec")
) == "Word2Vec"

# Beschreibung der Auswirkungen der Auswahl
if isWord2Vec:
    st.info(
        "Word2Vec vergleicht Wörter direkt und funktioniert am besten, wenn Ihre Anfrage spezifische Begriffe enthält. "
        "Dies führt zu präziseren Ergebnissen, wenn die genauen Begriffe in den Geschichten vorkommen."
    )
else:
    st.info(
        "Doc2Vec berücksichtigt den gesamten Kontext einer Geschichte und ist ideal, wenn Ihre Anfrage eher allgemein ist. "
        "Es liefert bessere Ergebnisse, wenn Sie keine genauen Begriffe haben."
    )

# Eingabefeld für die Anfrage
user_query = st.text_input("Geben Sie Ihre Anfrage ein:", placeholder="Beschreiben Sie Ihre Suche hier...")

# Button zum Suchen der ähnlichen Geschichten
if st.button("Ähnliche Geschichten finden"):
    if user_query.strip():
        st.write(f"Ihre Anfrage: '{user_query}' wird verarbeitet...")
        st.write(f"Verwendetes Modell: {'Word2Vec' if isWord2Vec else 'Doc2Vec'}")

        # Aufruf der Ranking-Funktion NACH dem Button-Klick
        rankings_df = main.rank_art_stories_python_function(user_query, isWord2Vec=isWord2Vec)

        # Ausgabe der Rankings als Tabelle
        if not rankings_df.empty:
            st.write("Ranking der Ähnlichkeiten pro Token:")
            st.table(rankings_df)
        else:
            st.warning("Keine Ähnlichkeiten gefunden.")
    else:
        st.warning("Bitte geben Sie eine gültige Anfrage ein.")
