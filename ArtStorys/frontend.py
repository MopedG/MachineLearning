import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

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

def plot_tsne(vectors, documents, visited_stories, recommendations):
    """
    Plots a t-SNE visualization of the ranking data.
    :param vectors: List of vectors to visualize.
    :param labels: List of document labels corresponding to the vectors.
    """
    if len(vectors) < 2:
        st.warning("Nicht genug Datenpunkte für die t-SNE Visualisierung.")
        return
    # Extract recommended filenames
    recommended_filenames = recommendations["artstory"].tolist()
    vectors = np.array(vectors)
    tsne = TSNE(n_components=2, random_state=24, perplexity=min(5, len(vectors-1)))
    vectors_2d = tsne.fit_transform(vectors)

    plt.clf()
    plt.figure(figsize=(12, 8))
    for i, doc in enumerate(documents):

        '''
        plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1], label=label if i < 10 else None, alpha=0.7)
        '''
        labels_added = {"Besucht": False, "Empfohlen": False, "Andere": False}

        x, y = vectors_2d[i]
        if i in visited_stories:
            plt.scatter(x, y, c="blue", label="Besucht" if not labels_added["Besucht"] else None, s=80,
                        edgecolors="black")
            labels_added["Besucht"] = True
        elif doc["filename"] in recommended_filenames:
            plt.scatter(x, y, c="red", label="Empfohlen" if not labels_added["Empfohlen"] else None, s=100,
                        edgecolors="black")
            labels_added["Empfohlen"] = True

        else:
            plt.scatter(x, y, c="gray", label="Andere" if not labels_added["Andere"] else None, alpha=0.5)
            labels_added["Andere"] = True

    # Add titles and labels
    plt.title("t-SNE Visualisierung der Art Story Cluster", fontsize=16)
    plt.xlabel("X-Achse", fontsize=12)
    plt.ylabel("Y-Achse", fontsize=12)
    plt.legend(bbox_to_anchor=(1,1.2), title='Besucht', loc= "upper left")
    # plt.colorbar()
    st.pyplot(plt)

# Streamlit Layout
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

query = st.session_state.get('query', None)

if query:
    st.write(f"Ihre Anfrage: '{query}' wird verarbeitet...")
    st.write(f"Verwendetes Modell: {'Word2Vec' if st.session_state['isWord2Vec'] else 'Doc2Vec'}")

# Ausgabe der Rankings als Tabelle
ranking = st.session_state.get('ranking', None)

if ranking is not None and not ranking.empty:
    st.write("Ranking der Ähnlichkeiten pro Token:")
    st.table(ranking)

    # Auswahl besuchter Geschichten
    st.markdown("#### Wählen Sie Geschichten aus, die Sie besucht haben:")
    visited_stories = []
    for idx, row in ranking.iterrows():
        if st.checkbox(f"Besucht: {row['Document']}", key=row['Document']):
            visited_stories.append(idx)

    st.session_state['visited_stories'] = visited_stories

    # Empfehlungen anzeigen
    if visited_stories:
        st.subheader("Empfohlene Geschichten basierend auf Ihren Besuchen:")
        result = main.recommend_art_stories_python_function(visited_stories)
        recommendations = result['recommendations']
        document_vectors = result['document_vectors']
        documents = result['documents']

        print(f"Besuchte Dokumente: {visited_stories}")
        print(f"Empfehlungen: {recommendations}")
        if not recommendations.empty:
            #st.write(recommendations)
            st.dataframe(recommendations)

            st.subheader("t-SNE Visualisierung der Empfehlungen und Cluster:")
            plot_tsne(document_vectors, documents, visited_stories, recommendations)
        else:
            st.warning("Keine Empfehlungen verfügbar.")

        print(f"Besuchte Dokumente: {visited_stories}")
        print(f"Empfehlungen: {recommendations}")

    else:
        st.warning("Bitte wählen Sie besuchte Geschichten aus.")

if ranking is not None:
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

def update_visited_stories(story_index):
    if 'visited_stories' not in st.session_state:
        st.session_state['visited_stories'] = []
    st.session_state['visited_stories'].append(story_index)
