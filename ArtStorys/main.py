import os
from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

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


def rank_art_stories_python_function(user_query_string):
    documents = documentToWord2Vec()

    for document in documents:
        document["model"].wv.similairty()

def test():
    documents = documentToWord2Vec()
    documents[0]['model'].wv.n_similarity()

if __name__ == "__main__":
    documentToWord2Vec()
