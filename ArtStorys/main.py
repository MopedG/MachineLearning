import os
from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import nltk


is_word2vec = True

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


'''
def get_vec_from_word_2_vec(input_txt, model):
    tokens = [word.lower() for word in word_tokenize(input_txt)]
    vectors = [model.wv[word] for word in tokens if word in model.wv.key_to_index]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)
'''


def rank_art_stories_python_function(query):

    if is_word2vec:
        # similarity_score_each_token(query)
        similarity_score_single_token(query)

    else:
        pass

# Dieser Ansatz vergleicht die Cosine Similarity jedes Tokens der Query, innerhalb der word2vec Models der einzelnen Artikel.
# Daraufhin wird die durchschnittliche Similarity der Tokens innerhalb jedes Artikels berechnet.
# Die Artikel werden dann nach der durchschnittlichen Similarity absteigend sortiert bzw. gerankt.
def similarity_score_each_token(query):
    documents = documentToWord2Vec()
    query_tokens = word_tokenize(query.lower())

    rankings = []
    for document in documents:
        similarities = []
        for i in range(len(query_tokens)):
            for j in range(i + 1, len(query_tokens)):
                token1 = query_tokens[i]
                token2 = query_tokens[j]
                if token1 in document["model"].wv and token2 in document["model"].wv:
                    similarity = document["model"].wv.similarity(token1, token2)
                    similarities.append(similarity)
                else:
                    similarities.append(0)

        if similarities:
            average_similarity = sum(similarities) / len(similarities)
        else:
            average_similarity = 0
        rankings.append((document["site"], average_similarity))

    rankings.sort(key=lambda x: x[1], reverse=True)

    # TODO: Return the rankings to the frontend !!! Only for Word2Vec
    for rank in rankings:
        print(f"Document: {rank[0]}, Similarity: {rank[1]}")

def similarity_score_single_token(query):
    documents = documentToWord2Vec()
    query_tokens = word_tokenize(query.lower())

    rankings = []
    for document in documents:

        similarities = []

        for token in query_tokens:

            if token in document["model"].wv:
                similarities.append(document["model"].wv.similarity(token, token))

        if similarities:
            average_similarity = sum(similarities) / len(similarities)
            print(average_similarity)
        else:
            average_similarity = 0
        rankings.append((document["site"], average_similarity))

    rankings.sort(key=lambda x: x[1], reverse=True)

    for rank in rankings:
        print(f"Document: {rank[0]}, Similarity: {rank[1]}")

if __name__ == "__main__":
    query = "hong kong architecture"
    rank_art_stories_python_function(query)