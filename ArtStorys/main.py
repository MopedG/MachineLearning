import os
from gensim.models import Word2Vec
import gensim
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

is_word2vec = True

def doc_2_vec(query: str):
    #nltk.download('punkt_tab')
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
    textfiles_folder = "C:\\Entw\\MachineLearning\\ArtStorys\\Textfiles"
    site_texts = []
    if os.path.exists(textfiles_folder):
        for filename in os.listdir(textfiles_folder):
            file_path = os.path.join(textfiles_folder, filename)
            if os.path.isfile(file_path):
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

if __name__ == "__main__":
    query = "cathedral fire france excavation notre-dame reconstruction"
    print("query: ", query)
    rank_art_stories_python_function(query)