import streamlit as st
import main

st.title("Ranking von Art Stories")
st.subheader("Finden Sie die ähnlichsten Geschichten zu Ihrer Anfrage")

# Eingabefeld für die Anfrage
user_query = st.text_input("Geben Sie Ihre Anfrage ein:", placeholder="Beschreiben Sie Ihre Suche hier...")

# Button zum Suchen der ähnlichen Geschichten
if st.button("Ähnliche Geschichten finden"):
    if user_query.strip():
        st.write(f"Ihre Anfrage: '{user_query}' wird verarbeitet...")

        # Aufruf der Ranking-Funktion NACH dem Button-Klick
        rankings = main.rank_art_stories_python_function(user_query)

        # Ausgabe der Rankings als Tabelle
        if rankings:
            st.write("Ranking der Ähnlichkeiten pro Token:")

            # Streamlit-Komponente zum Anzeigen als Tabelle
            st.table(rankings)
        else:
            st.warning("Keine Ähnlichkeiten gefunden.")
    else:
        st.warning("Bitte geben Sie eine gültige Anfrage ein.")

