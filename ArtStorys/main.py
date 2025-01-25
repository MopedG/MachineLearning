import os
from itertools import count

from gensim.models import Word2Vec
import gensim
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk import corpus
from nltk.parse.util import taggedsents_to_conll
from nltk.tokenize import sent_tokenize, word_tokenize
from bertopic import BERTopic
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE ## NEW
import matplotlib.pyplot as plt ## NEW

is_word2vec = True

def train_doc2vec_model(documents):
    formatted_documents = [doc["content"].lower().replace("-", " ") for doc in documents]
    tagged_document_data = [TaggedDocument(words=word_tokenize(doc), tags=[str(i)]) for i, doc in enumerate(formatted_documents)]

    model = Doc2Vec(tagged_document_data, min_count=2, vector_size=100, epochs=50)
    model.build_vocab(tagged_document_data)
    model.train(tagged_document_data, total_examples=model.corpus_count, epochs=model.epochs)

    return model

def doc_2_vec(query: str):
    nltk.download('punkt_tab')
    documents = filesToStrings()
    formatted_query = query.replace('-', ' ')
    formatted_documents = list(map(lambda site_text: site_text['content'].lower().replace('-', ' '), documents))

    tagged_data = [
        TaggedDocument(
            words=word_tokenize(document), # case for covering dashes inbetween words (e.g. notre-dame)
            tags=[str(i)]
        )
            for i, document in enumerate(formatted_documents)
    ]

    model = Doc2Vec(tagged_data, min_count=2, vector_size=100, epochs=50)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    user_input_vector = model.infer_vector(word_tokenize(formatted_query))
    document_vectors = [
        model.infer_vector(
            word_tokenize(document)
        )
            for document in formatted_documents
    ]
    similarities = model.wv.cosine_similarities(user_input_vector, document_vectors)
    ranking = []
    for i in range(len(documents)):
        ranking.append(
            {
                'document': documents[i]["filename"],
                'similarity': similarities[i]
            }
        )
    ranking.sort(key=lambda x: x['similarity'], reverse=True)

    return pd.DataFrame({
        'artstory': (rank['document'] for rank in ranking),
        'similarity': (rank['similarity'] for rank in ranking)
    })

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

def rank_art_stories_python_function(query, isWord2Vec):
    return similarity_score_query_to_article_with_idf(query) if isWord2Vec else doc_2_vec(query)


# Dieser Ansatz trainiert ein Word2Vec Model auf dem gesamten Korpus und repräsentiert jeden Artikel als Vektor (Durchschnitt der Wortvektoren).
# Daraufhin wird die query als Vektor eingebettet und mit den Artikel-Vektoren verglichen.
def similarity_score_query_to_article(query):
    print("Hallo?")
    documents = filesToStrings()
    query_tokens = word_tokenize(query.lower())

    all_sentences = []
    for document in documents:
        all_sentences.extend([word_tokenize(sent.lower()) for sent in sent_tokenize(document["content"])])

    model = gensim.models.Word2Vec(all_sentences, min_count=1, vector_size=100, window=5)

    document_vectors = []
    for document in documents:
        tokens = word_tokenize(document["content"].lower())
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        if vectors:
            document_vector = np.mean(vectors, axis=0)
        else:
            document_vector = np.zeros(model.vector_size)
        document_vectors.append((document["filename"], document_vector))

    query_vectors = [model.wv[token] for token in query_tokens if token in model.wv]
    if query_vectors:
        query_vector = np.mean(query_vectors, axis=0)
    else:
        query_vector = np.zeros(model.vector_size)

    # Compare the query vector to the article embeddings using cosine similarity
    df_rankings = pd.DataFrame(columns=['Document', 'Similarity'])
    for site, doc_vector in document_vectors:
        similarity = cosine_similarity([query_vector], [doc_vector])[0][0]
        df_rankings = df_rankings.append({'Document': site, 'Similarity': similarity}, ignore_index=True)

    df_rankings = df_rankings.sort_values(by='Similarity', ascending=False)

    for index, row in df_rankings.iterrows():
        print(f"Document: {row['Document']}, Similarity: {row['Similarity']}")

    return df_rankings


def similarity_score_query_to_article_with_idf(query):
    documents = filesToStrings()
    corpus = [doc["content"] for doc in documents]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([query])

    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    df_rankings = pd.DataFrame(columns=['Document', 'Similarity'])
    for i, score in enumerate(similarity_scores):
        temp_df = pd.DataFrame({'Document': [documents[i]["filename"]], 'Similarity': [score]})
        df_rankings = pd.concat([df_rankings, temp_df], ignore_index=True)

    df_rankings = df_rankings.sort_values(by='Similarity', ascending=False)
    for index, row in df_rankings.iterrows():
        print(f"Document: {row['Document']}, Similarity: {row['Similarity']}")

    return df_rankings


"""
user_ranking: e.g. [1, 0, None, 3, 2]
"""
def map_at_k(k, user_ranking):
    map_at_k_sum = 0
    for i in range(k):
        map_at_k_sum += sum(1 for x in user_ranking[:(i + 1)] if x is not None and x <= i) / (i + 1)

    return map_at_k_sum / k


"""
visited_stories: indices for visited stories, e.g. [1, 2, 5]
"""
def recommend_art_stories_python_function(visited_stories):
    ## Recommends the top-3 art stories based on the visited
    ## @param visited_stories: List of visited art stories
    ## @return: DataFrame with top-3 recommendations for next documents
    documents = filesToStrings()
    corpus = [doc["content"] for doc in documents]
    doc2vec_model = train_doc2vec_model(documents)

    # Calculate document vectors
    document_vectors = [doc2vec_model.infer_vector(word_tokenize(doc["content"])) for doc in documents]
    visited_vectors = [document_vectors[i] for i in visited_stories]

    # Calculate average vector for visited stories
    avg_vector = np.mean(visited_vectors, axis=0)

    # Calculate cosine similarity between average vector and all document vectors
    similarities = cosine_similarity([avg_vector], document_vectors).flatten()

    # Sort documents by similarity
    recommendations = [
        {"artstory": documents[i]["filename"], "similarity": similarities[i]}
        for i in range(len(documents)) if i not in visited_stories
    ]
    recommendations = sorted(recommendations, key=lambda x: x["similarity"], reverse=True)[:3]

    # Prepare data for clustering
    labels = {
        doc["filename"]: {"filename": doc["filename"], "is_visited": i in visited_stories, "is_recommended": False}
        for i, doc in enumerate(documents)
    }
    # Mark recommended documents
    for rec in recommendations:
        if rec["artstory"] in labels:
            labels[rec["artstory"]]["is_recommended"] = True

    recommendation_df = pd.DataFrame(recommendations)
    print(recommendation_df)

    '''
    # Optional: t-SNE-Visualisierung
    plot_tsne(document_vectors, [doc["filename"] for doc in documents])
    '''

    return {"recommendations": recommendation_df, "document_vectors": document_vectors, "documents": documents}
if __name__ == "__main__":
    ## TEST FOR DEBUGGING IN BACKEND
    query = "cathedral fire france excavation notre-dame reconstruction"
    print("query: ", query)
    #rank_art_stories_python_function(query, isWord2Vec=False)
    recommended = recommend_art_stories_python_function(visited_stories=[3, 2, 4])