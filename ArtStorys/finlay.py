import os
from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize

# Converts a (document) string to a model for Word2Vec Embedding
def documentToWord2Vec():
    site_texts = filesToStrings()
    document_models = []
    for site_text in site_texts:
        data = []
        for i in sent_tokenize(site_text["content"]):
            temp = []

            for j in word_tokenize(i):
                temp.append(j.lower())

            data.append(temp)

        model = gensim.models.Word2Vec(data, min_count=1, vector_size=100, window=5)

        document_models.append(
            {
                "site": site_text["filename"],
                "content": site_text["content"],
                "model": model
            }
        )

    return document_models



def tokenizeText(text):
    data = []
    for i in sent_tokenize(text):
        temp = []

        for j in word_tokenize(i):
            temp.append(j.lower())

        data.append(temp)


def filesToStrings():
    textfiles_folder = "./Textfiles"
    site_texts = []
    # Überprüfen, ob der Ordner existiert
    if os.path.exists(textfiles_folder):
        # Alle Dateien im Ordner durchlaufen
        for filename in os.listdir(textfiles_folder):
            file_path = os.path.join(textfiles_folder, filename)
            # Überprüfen, ob es sich um eine Datei handelt
            if os.path.isfile(file_path):
                # Datei öffnen und Inhalt lesen
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    site_texts.append(
                        {
                            "filename": filename,
                            "content": content.replace("\n", " ")
                        }
                    )

    return site_texts


def rank_art_stories_python_function(query):
    documents = documentToWord2Vec()
    query_tokens = word_tokenize(query.lower())

    query_vectors = []
    for token in query_tokens:
        if token in documents[0]["model"].wv:
            query_vectors.append(documents[0]["model"].wv[token])

    query_vector = sum(query_vectors) / len(query_vectors)

    rankings = []
    for document in documents:
        document_vectors = [document["model"].wv[word] for word in word_tokenize(document["content"].lower()) if
            word in document["model"].wv]
        similarities = []

        for doc_vector in document["model"]:
            # Berechne die Cosinus-Ähnlichkeit zwischen dem Dokumenten-Vektor und dem Anfrage-Vektor
            similarities.append(document["model"].wv.similarity(query_vector))

            # Berechne den Durchschnitt der Ähnlichkeiten für das Dokument
        average_similarity = sum(similarities) / len(similarities)
        rankings.append((document["site"], average_similarity))

        #for token in query_tokens:

            #if token in document["model"].wv:

                #similarities.append(document["model"].wv.similarity(query_vector))

        #if similarities:
            #average_similarity = sum(similarities) / len(similarities)
            #print(average_similarity)
        #else:
            #average_similarity = 0
        #rankings.append((document["site"], average_similarity))

    rankings.sort(key=lambda x: x[1], reverse=True)

    for rank in rankings:
        print(f"Document: {rank[0]}, Similarity: {rank[1]}")


if __name__ == "__main__":
    query = "architecture notre-dame"
    rank_art_stories_python_function(query)