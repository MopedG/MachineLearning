import streamlit as st
import main

def map_at_k_input_validation(k, user_ranking):
    if k.strip() == "":
        return "K darf nicht leer sein."

    try:
        parsed_k = int(k)
        parsed_user_ranking = list(
            map(lambda x: None if str(x).strip() == "" else int(x), user_ranking)
        )
    except:
        return "Einige Angaben sind keine Ganzzahlen."

    if parsed_k <= 0:
        return "K muss positiv sein"

    if parsed_k > len(user_ranking):
        return f"K darf nicht größer als {len(user_ranking)} sein."

    if len(user_ranking) != len(set(user_ranking)):
        return "Dein Ranking darf keine Duplikate enthalten."

    for item in parsed_user_ranking:
        if item is not None and item > parsed_k - 1:
            return f"Ein Ranking darf maximal den Index {parsed_k - 1} besitzen."

    return None



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
            st.session_state['query'] = user_query.strip()
            st.session_state['isWord2Vec'] = isWord2Vec
            # Aufruf der Ranking-Funktion NACH dem Button-Klick
            st.session_state['ranking'] = main.rank_art_stories_python_function(user_query, isWord2Vec=isWord2Vec)

        else:
            st.warning("Bitte geben Sie eine gültige Anfrage ein.")


query = None
try:
    query = st.session_state['query']
except:
    pass

if query:

    isWord2Vec_session_state = None
    try:
        isWord2Vec_session_state = st.session_state['isWord2Vec']
    except:
        pass

    st.write(f"Ihre Anfrage: '{query}' wird verarbeitet...")
    st.write(f"Verwendetes Modell: {'Word2Vec' if isWord2Vec_session_state else 'Doc2Vec'}")

# Ausgabe der Rankings als Tabelle

ranking = None
try:
    ranking = st.session_state['ranking']
except:
    pass


if ranking is not None:
    if not ranking.empty:
        st.write("Ranking der Ähnlichkeiten pro Token:")
        st.table(st.session_state['ranking'])

        st.markdown("#### Berechne MAP@K")
        user_ranking = {}
        col1, col2 = st.columns([1, 10])
        for i in range(len(ranking)):
            col1.text_input(label=str(i), label_visibility="hidden", key=str(i) + "indicator", disabled=True, value=str(i))
            user_ranking[i] = col2.text_input(label=str(i), label_visibility="hidden", key=i)

        k = st.text_input(label="K", placeholder="K", value=str(len(ranking)))

        if st.button('Berechne MAP@K'):
            err_msg = map_at_k_input_validation(k, user_ranking.values())

            if err_msg:
                st.error(err_msg)
            else:
                map_at_k = main.map_at_k(int(k), list(map(lambda x: None if x.strip() == "" else int(x), user_ranking.values())))
                st.markdown(f"MAP@K (K={k}): `{map_at_k}`")
    else:
        st.warning("Keine Ähnlichkeiten gefunden.")