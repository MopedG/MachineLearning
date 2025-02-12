import os
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


is_word2vec = True

"""
Trainiert das Doc2Vec-Modell auf den gegebenen Dokumenten mit 50 Epochen und gibt das trainierte Modell zurück
"""
def train_doc2vec_model(documents):
    formatted_documents = [doc["content"].lower().replace("-", " ") for doc in documents]
    tagged_document_data = [TaggedDocument(words=word_tokenize(doc), tags=[str(i)]) for i, doc in enumerate(formatted_documents)]

    model = Doc2Vec(tagged_document_data, min_count=2, vector_size=100, epochs=50)
    model.build_vocab(tagged_document_data)
    model.train(tagged_document_data, total_examples=model.corpus_count, epochs=model.epochs)

    return model

"""
Berechnet die Ähnlichkeiten zwischen der Suchanfrage und den Dokumenten und gibt die Ergebnisse zurück
"""
def doc_2_vec(query: str):
    nltk.download('punkt_tab')  # Nötiges NLTK-Modul für Tokenisierung herunterladen (muss nur einmal ausgeführt werden)
    documents = filesToStrings()
    formatted_query = query.replace('-', ' ')
    formatted_documents = list(map(lambda site_text: site_text['content'].lower().replace('-', ' '), documents))

    # Dokumente in tokenisierte Wörter umwandeln und mit einer ID versehen
    tagged_data = [
        TaggedDocument(
            words=word_tokenize(document),
            tags=[str(i)]
        )
        for i, document in enumerate(formatted_documents)
    ]

    # Doc2Vec-Modell initialisieren und trainieren
    model = Doc2Vec(tagged_data, min_count=2, vector_size=100, epochs=50)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    # Vektor für die Suchanfrage berechnen
    user_input_vector = model.infer_vector(word_tokenize(formatted_query))

    # Vektoren für die Dokumente berechnen
    document_vectors = [
        model.infer_vector(word_tokenize(document))
        for document in formatted_documents
    ]

    # Ähnlichkeiten zwischen der Anfrage und den Dokumenten berechnen
    similarities = model.wv.cosine_similarities(user_input_vector, document_vectors)

    # Erstelle Ranking
    ranking = []
    for i in range(len(documents)):
        ranking.append(
            {
                'document': documents[i]["filename"],
                'similarity': similarities[i]
            }
        )

    # Ergebnisse nach Ähnlichkeit sortieren
    ranking.sort(key=lambda x: x['similarity'], reverse=True)

    return pd.DataFrame({
        'artstory': (rank['document'] for rank in ranking),
        'similarity': (rank['similarity'] for rank in ranking)
    })

"""
Holt die Dokumente und gibt sie als eine Liste von Strings zurück
"""
def filesToStrings():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    textfiles_folder = os.path.join(current_directory, 'Textfiles')
    site_texts = []
    if os.path.exists(textfiles_folder):
        for filename in os.listdir(textfiles_folder):
            file_path = os.path.join(textfiles_folder, filename)
            if os.path.isfile(file_path) and file_path.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    site_texts.append(
                        {
                            "filename": filename,
                            "content": content.replace("\n", " ")
                        }
                    )
    return site_texts

"""
Führt die gesamte Logik für die Ranking-Funktion aus
"""
def rank_art_stories_python_function(query, isWord2Vec):
    return word_2_vec(query) if isWord2Vec else doc_2_vec(query)

"""
Durchsucht die Texte basierend auf der Suchanfrage und gibt die Ergebnisse zurück
"""
def word_2_vec(query):
    # Lädt die gespeicherten Dokumente als Strings
    documents = filesToStrings()

    # Speichert den Inhalt der Dokumente in einer Liste.
    corpus = [doc["content"] for doc in documents]

    # Der corpus wird verwendet um eine tfidf Matrix zu erstellen
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Wandelt die Suchanfrage ebenfalls in einen TF-IDF-Vektor um
    query_vector = vectorizer.transform([query])

    # Berechnet die Ähnlichkeiten zwischen den Vektor der Suchanfrage und der Matrix des Corpus', mittels Cosine Similarity
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Erstellt ein DataFrame mit den Ergebnissen und sortiert sie nach der Ähnlichkeit
    df_rankings = pd.DataFrame(columns=['Document', 'Similarity'])
    for i, score in enumerate(similarity_scores):
        temp_df = pd.DataFrame({'Document': [documents[i]["filename"]], 'Similarity': [score]})
        df_rankings = pd.concat([df_rankings, temp_df], ignore_index=True)

    df_rankings = df_rankings.sort_values(by='Similarity', ascending=False)
    for index, row in df_rankings.iterrows():
        print(f"Document: {row['Document']}, Similarity: {row['Similarity']}")

    return df_rankings


"""
Berechnet den Mean Average Precision (MAP) at k für die gegebenen Werte
user_ranking: e.g. [1, 0, None, 3, 2]
"""
def map_at_k(k, user_ranking):
    map_at_k_sum = 0
    for i in range(k):
        map_at_k_sum += sum(1 for x in user_ranking[:(i + 1)] if x is not None and x <= i) / (i + 1)

    return map_at_k_sum / k



"""
Empfehlungsfunktion, die die Top-3-Kunstgeschichten basierend auf den besuchten Kunstgeschichten empfiehlt
visited_stories: indices for visited stories, e.g. [1, 2, 5]
rückgabe: DataFrame mit den Top-3-Empfehlungen für die nächsten Dokumente
"""
def recommend_art_stories_python_function(visited_stories):
    documents = filesToStrings()
    corpus = [doc["content"] for doc in documents]
    doc2vec_model = train_doc2vec_model(documents)

    # Berechnet die Vektoren für die Dokumente
    document_vectors = [doc2vec_model.infer_vector(word_tokenize(doc["content"])) for doc in documents]
    visited_vectors = [document_vectors[i] for i in visited_stories]

    # Berechnet den Durchschnittsvektor für die besuchten Geschichten
    avg_vector = np.mean(visited_vectors, axis=0)

    # Berechnet die Kosinusähnlichkeit zwischen dem Durchschnittsvektor und allen Dokumentenvektoren
    similarities = cosine_similarity([avg_vector], document_vectors).flatten()

    # Sortiert die Dokumente nach Ähnlichkeit
    recommendations = [
        {"artstory": documents[i]["filename"], "similarity": similarities[i]}
        for i in range(len(documents)) if i not in visited_stories
    ]
    recommendations = sorted(recommendations, key=lambda x: x["similarity"], reverse=True)[:3]

    # Vorbereitung der Daten für die Visualisierung
    labels = {
        doc["filename"]: {"filename": doc["filename"], "is_visited": i in visited_stories, "is_recommended": False}
        for i, doc in enumerate(documents)
    }
    # Markiere empfohlene Dokumente
    for rec in recommendations:
        if rec["artstory"] in labels:
            labels[rec["artstory"]]["is_recommended"] = True

    # Erstelle DataFrame für die Empfehlungen
    recommendation_df = pd.DataFrame(recommendations)
    print(recommendation_df)
    return {"recommendations": recommendation_df, "document_vectors": document_vectors, "documents": documents}

if __name__ == "__main__":
    ## TEST FOR DEBUGGING IN BACKEND
    query = "cathedral fire france excavation notre-dame reconstruction"
    print("query: ", query)
    # Test ranking function
    recommended = recommend_art_stories_python_function(visited_stories=[3, 2, 4])