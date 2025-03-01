import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

import main

# Mapping von Art Storys zu Ranking-Indizes
def map_artstory_to_ranking_index(artstory_as_string, ranking):
    for i, item in enumerate(ranking["Document"]):
        if item == artstory_as_string:
            return i
    return None

# Map@K Eingabevalidierung
def map_at_k_input_validation(k, user_ranking):
    if k.strip() == "":
        return "K darf nicht leer sein."

    try:
        parsed_k = int(k)
    except:
        return "K muss eine Ganzzahl sein."

    if parsed_k <= 0:
        return "K muss positiv sein."

    if parsed_k > len(user_ranking):
        return f"K darf nicht größer als {len(user_ranking)} sein."

    non_none_user_ranking = list(filter(lambda x: x is not None, user_ranking))
    if len(non_none_user_ranking) != len(set(non_none_user_ranking)):
        return "Dein Ranking darf keine Duplikate enthalten."

    for item in user_ranking:
        if item is not None and item > parsed_k - 1:
            return f"Ein Ranking darf maximal den Index {parsed_k - 1} besitzen."

    return None

# Plottet eine t-SNE Visualisierung der Ranking-Daten
# vectors: Liste von Vektoren, die visualisiert werden sollen
# labels: Liste von Dokumentenlabels, die den Vektoren entsprechen
def plot_tsne(vectors, documents, visited_stories, recommendations):
    if len(vectors) < 2:
        st.warning("Nicht genug Datenpunkte für die t-SNE Visualisierung.")
        return
    # Vorgeschlagene Dateinamen extrahieren
    recommended_filenames = recommendations["artstory"].tolist()
    vectors = np.array(vectors)
    # t-SNE Konfiguration
    tsne = TSNE(n_components=2, random_state=24, perplexity=min(5, len(vectors-1)))
    vectors_2d = tsne.fit_transform(vectors)

    # Plot
    plt.clf()
    plt.figure(figsize=(12, 8))
    for i, doc in enumerate(documents):
        labels_added = {"Besucht": False, "Empfohlen": False, "Andere": False}
        # Farben für besuchte, empfohlene und andere Dokumente
        x, y = vectors_2d[i]
        if i in visited_stories:
            plt.scatter(x, y, c="blue", label="Besucht" if not labels_added["Besucht"] else None, s=80,
                        edgecolors="black")
            labels_added["Besucht"] = True
        elif doc["filename"] in recommended_filenames:
            plt.scatter(x, y, c="red", label="Empfohlen" if not labels_added["Empfohlen"] else None, s=100,
                        edgecolors="black")
            labels_added["Empfohlen"] = True
        # Sonstige Dokumente in grau
        else:
            plt.scatter(x, y, c="gray", label="Andere" if not labels_added["Andere"] else None, alpha=0.5)
            labels_added["Andere"] = True

    # Hinzufügen von Beschriftungen und Labels
    plt.title("t-SNE Visualisierung der Art Story Cluster", fontsize=16)
    plt.xlabel("X-Achse", fontsize=12)
    plt.ylabel("Y-Achse", fontsize=12)
    plt.legend(bbox_to_anchor=(1,1.2), title='Besucht', loc= "upper left")
    st.pyplot(plt)

# Streamlit Beginn
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
# Ranking anzeigen
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
        if not recommendations.empty:
            st.write(recommendations)
            st.subheader("t-SNE Visualisierung der Empfehlungen und Cluster:")
            plot_tsne(document_vectors, documents, visited_stories, recommendations)
        else:
            st.warning("Keine Empfehlungen verfügbar.")

        print(f"Besuchte Dokumente: {visited_stories}")
        print(f"Empfehlungen: {recommendations}")

    else:
        st.warning("Bitte wählen Sie besuchte Geschichten aus.")

# Berechnung von MAP@K
if ranking is not None:
    st.markdown("#### Berechne MAP@K")
    st.info("""
    Gib' hier das Ranking ein, was **du** für richtig hälst. Der MAP@K-Wert sagt aus, wie gut das
    berechnete Ranking mit deinem übereinstimmt.  
    0 = Keine Übereinstimmung  
    1 = Voll Übereinstimmung
    """)

    user_ranking = {}
    col1, col2 = st.columns([1, 10])

    # Eigenschaften für Eingabefelder des Rankings
    options = [""]
    for item in ranking["Document"]:
        options.append(item)
    for i in range(len(ranking)):
        col1.text_input(label=str(i), label_visibility="hidden", key=str(i) + "indicator", disabled=True, value=str(i))
        user_ranking[i] = col2.selectbox(label=str(i), label_visibility="hidden", key=i, options=options)

    k = st.text_input(label="K", placeholder="K", value=str(len(ranking)))

    # Berechnung von MAP@K durchführen
    if st.button('Berechne MAP@K'):
        user_ranking_as_indices = [map_artstory_to_ranking_index(x, ranking) for x in user_ranking.values()]
        err_msg = map_at_k_input_validation(k, user_ranking_as_indices)

        if err_msg:
            st.error(err_msg)
        else:
            map_at_k = main.map_at_k(int(k), user_ranking_as_indices)
            st.markdown(f"MAP@K (K={k}): `{map_at_k}`")
else:
    st.warning("Keine Ähnlichkeiten gefunden.")

# Aktualisieren der besuchten Geschichten
def update_visited_stories(story_index):
    if 'visited_stories' not in st.session_state:
        st.session_state['visited_stories'] = []
    st.session_state['visited_stories'].append(story_index)
