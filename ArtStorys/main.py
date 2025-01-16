import os
from gensim.models import Word2Vec
import gensim
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

is_word2vec = True

def doc_2_vec(site_texts, user_input: str):
    nltk.download('punkt_tab')
    tagged_data = [
        TaggedDocument(
            words=word_tokenize(doc['content'].lower().replace('-', ' ')), # case for covering dashes inbetween words (e.g. notre-dame)
            tags=[str(i)]
        )
            for i, doc in enumerate(site_texts)
    ]

    model = Doc2Vec(tagged_data, min_count=2, vector_size=100, epochs=50)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    user_input_vector = model.infer_vector(word_tokenize(user_input.lower()))

    document_vectors = [
        model.infer_vector(
            word_tokenize(doc['content'].lower())
        )
            for doc in site_texts
    ]

    similarities = model.wv.cosine_similarities(user_input_vector, document_vectors)

    ranking = []
    for i in range(len(site_texts)):
        ranking.append(
            {
                'document': site_texts[i],
                'similarity': similarities[i]
            }
        )

    ranking.sort(key=lambda x: x['similarity'], reverse=True)
    return ranking




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
    textfiles_folder = "C:\\Entw\\MachineLearning\\ArtStorys\\Textfiles"
    site_texts = []
    # Überprüfen, ob der Ordner existiert
    if os.path.exists(textfiles_folder):
        # Alle Dateien im Ordner durchlaufen
        print(f"Der Ordner {textfiles_folder} existiert")
        for filename in os.listdir(textfiles_folder):
            print(f"Lese Datei: {filename}")
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
        # Rufe die Methode auf und speichere die Rankings in einer Variable
        rankings = similarity_score_query_to_article(query)
        return rankings
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

# Dieser Ansatz trainiert ein Word2Vec Model auf dem gesamten Korpus und repräsentiert jeden Artikel als Vektor (Durchschnitt der Wortvektoren).
# Daraufhin wird die query als Vektor eingebettet und mit den Artikel-Vektoren verglichen.
def similarity_score_query_to_article(query):
    documents = documentToWord2Vec()
    query_tokens = word_tokenize(query.lower())

    # Train a Word2Vec model on the entire corpus
    all_sentences = []
    for document in documents:
        all_sentences.extend([word_tokenize(sent.lower()) for sent in sent_tokenize(document["content"])])

    model = gensim.models.Word2Vec(all_sentences, min_count=1, vector_size=100, window=5)

    # Represent each article as a vector (average of word vectors)
    document_vectors = []
    for document in documents:
        tokens = word_tokenize(document["content"].lower())
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        if vectors:
            document_vector = np.mean(vectors, axis=0)
        else:
            document_vector = np.zeros(model.vector_size)
        document_vectors.append((document["site"], document_vector))

    # Embed the query using the same Word2Vec model
    query_vectors = [model.wv[token] for token in query_tokens if token in model.wv]
    if query_vectors:
        query_vector = np.mean(query_vectors, axis=0)
    else:
        query_vector = np.zeros(model.vector_size)

    # Compare the query vector to the article embeddings using cosine similarity
    rankings = []
    for site, doc_vector in document_vectors:
        similarity = cosine_similarity([query_vector], [doc_vector])[0][0]
        rankings.append((site, similarity))

    # Rank the articles based on similarity scores
    rankings.sort(key=lambda x: x[1], reverse=True)

    # Gib die Rankings als Liste zurück
    return rankings

if __name__ == "__main__":
    query = "traces polychrome"
    rank_art_stories_python_function(query)