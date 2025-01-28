"""
Prüfungsaufgabe 3
Programmieren Sie ein Wissensgraph Query Interface für ein Graph Neural Network der auf Moodle bereitgestellten
Kunstshow-Ontologie als Streamlit-App für die Abfrage durch einen Kunstshow-Experten.

a) App-Aufbau
    1 User Query: Anfrage (=Prompt) in natürlicher Sprache.
    2 CNF Expression: Ausgabe in konjunktiver Normalform.
    3 Highlighted Subgraph: Visueller Fokus auf die relevanten Knoten und Kanten.
    4 SPARQL Query: Abfrage des RDF-Graphen.
    5 Missing Node Classification: vorhergesagte fehlende binäre Knotenklasse VIPStatus ja/nein.
    6 Query Response: Ergebnis der Abfrage.
    - Tipp: Beginnen Sie mit wenigen Kontakten und Person-Accounts.
b) Erweitern Sie die Streamlit-App um den Anwendungsfall der zielgruppenorientierten Ansprache
von ähnlichen Kontakten und Person-Accounts (=Personen, die keiner Firma zugeordnet werden
können) für Newsletter Marketing-Kampagnen.
    - Beispiel-Anfrage: "Welche Kontakte können beim Versand meiner
      Newsletter Marketing-Kampagne zu einer Zielgruppe zusammengefasst
      werden?"
    - Tipp: Beginnen Sie mit wenigen Kontakten und Person-Accounts.
"""

"""
ERKENNTNISSE: NLP mit Spacy ist sehr unregelmäßig, d.h. es ist schwer, ein allgemeines zu finden, die Fälle dynamisch abdeckt.
- Es ist besser, die Entitäten und Schlüsselwörter entweder zum Teil dynamisch, zum Teil manuel zu extrahieren und dann die CNF-Formel zu generieren.
- Wir haben uns jedoch dazu entschlossen, einige Beispiel Queries zu definieren, die wir dann in CNF umwandeln.
"""

import spacy
import rdflib
from markdown_it.common.entities import entities
from sympy import symbols
from sympy.logic.boolalg import And, Or, Not, to_cnf

nlp = spacy.load("de_core_news_sm") # German model
#nlp = spacy.load("en_core_web_sm") # English model

testquery = "At what universities do the Turing Award winners in the field of Deep Learning work?"
query1 = "Welche user haben tickets für die veranstaltung 'Art Basel'?"
query2 = "Welches Artwork hat den Titel ""Golden Statue""?"

query2 = "How many tickets exist for the event 'Art Basel'?"
query3 = "Welche Tickets haben den TicketType ""Premium Ticket""?"
query4 = "Welche PersonAccounts kommen aus dem BillingState 'USA'?"

queryAufbau = "WELCHE X  HAT Y "
X = "X: bspw. PersonAccount, Contact, Artwork, Ticket, User, Account, Event"
Y = "Y: bspw. Name, Title, TicketType, BillingState, EventName, AccountName"

dict = {
    "query1": "Aus welchem Land kommt der Account 'Wealth Management AG'?",
    "query2": "Welches Artwork hat den Titel 'Golden Statue'?",
    "query3": "Welche ArtworkMedium haben den artMediumTitle 'Photography'?",
    "query4": "Welche Tickets haben den TicketType 'Premium Ticket'?",
    "query5": "Welche PersonAccounts kommen aus dem BillingState 'USA'?",
    "query6": "Welche Contacts haben eine mail unter '@test.com'?"
}

for key, value in dict.items():
    print(f"Key: {key}, Value: {value}")


doc = nlp(testquery)
entities = [(ent.text, ent.label_) for ent in doc.ents]
print("Gefundene Entitäten:", entities)

entities.append(("Turing Award", "AWARD"))
entities.append(("Deep Learning", "FIELD"))

# Define a function to process natural language query and generate CNF
V = symbols("V")
U = symbols("U")
TuringAward = symbols("TuringAward")
DeepLearning = symbols("DeepLearning")


for token in doc:
    if token.pos_ == "NOUN" or token.pos_ == "PROPN":
        pass# print(token.text)

# Define a function to process natural language query and generate CNF
def process_query_to_cnf(query: str) -> str:
    doc = nlp(query)

    # Extract all named entities and relationships dynamically
    variables = []
    conditions = []

    for ent in doc.ents:
        variables.append(ent.text)

    for token in doc:
        if token.pos_ == "VERB" or token.dep_ in {"ROOT"}:
            conditions.append(token.lemma_)

    # Dynamically generate CNF expressions based on extracted entities and conditions
    cnf_parts = []
    for i, var in enumerate(variables):
        if i == 0:
            cnf_parts.append(f"{conditions[0]}({var}, V)")
        else:
            cnf_parts.append(f"condition_{i}({var}, V)")

    if cnf_parts:
        cnf = f"q = U? . EV: {' ∧ '.join(cnf_parts)} ∧ target(V, U?)"
        return cnf
    return "Unable to process query into CNF."

print(process_query_to_cnf(testquery))

#print(f"Query: {query}")
#print([w.text, w.pos_] for w in doc)
#print([(ent.text, ent.label_) for ent in doc.ents])

#entities = [(ent.text, ent.label_) for ent in doc.ents]
#print(f"Entities: {entities}")
#tokens = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN", "VERB")]

#entities = [(ent.text, ent.label_) for ent in doc.ents]
tokens = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN", "VERB")]

# Extrahiere wichtige Teile der Anfrage
entities = [(ent.text, ent.label_) for ent in doc.ents]  # Named Entities
keywords = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN", "VERB", "ADJ")]

print("Gefundene Entitäten:", entities)
print("Gefundene Schlüsselwörter:", keywords)
#print("Wichtige Schlüsselwörter:", tokens)

