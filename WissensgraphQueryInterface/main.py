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
import os

"""
ERKENNTNISSE: NLP mit Spacy ist sehr unregelmäßig, d.h. es ist schwer, ein allgemeines zu finden, die Fälle dynamisch abdeckt.
- Es ist besser, die Entitäten und Schlüsselwörter entweder zum Teil dynamisch, zum Teil manuel zu extrahieren und dann die CNF-Formel zu generieren.
- Wir haben uns jedoch dazu entschlossen, einige Beispiel Queries zu definieren, die wir dann in CNF umwandeln.
"""

import ollama
from owlready2 import get_ontology, sync_reasoner, default_world
from rdflib import Graph

dict = {
    "query1": "Aus welchem Land kommt der Account 'Wealth Management AG'?",
    "query2": "Welches Artwork hat den Titel 'Golden Statue'?",
    "query3": "Welche ArtworkMedium haben den artMediumTitle 'Photography'?",
    "query4": "Welche Tickets haben den TicketType 'Premium Ticket'?",
    "query5": "Welche PersonAccounts kommen aus dem BillingState 'USA'?",
    "query6": "Welche Contacts haben eine mail unter '@test.com'?"
}


current_directory = os.path.dirname(os.path.abspath(__file__))
ontology_file = os.path.join(current_directory, 'art_show_ontology.ttl')
print(ontology_file)

graph = Graph()
graph.parse(ontology_file, format="ttl")


res = graph.query(
    """
    SELECT ?accountName WHERE
    {            
        ?account a :Account ;
                 :accountName ?accountName .
        ?contact a :Contact ;
                 :contactFullName ?contactFullName ;
                 :employedBy ?account .            
        FILTER(?contactFullName = "Henry Smith")
    }
    """
)
#
print(len(res))
for row in res:
    print(f"{row['accountName']}")

# Welcher Account hat welches Artwork gekauft?
# Account gekauft Kunstwerk?
# X , Y , Z: X: Account, Y: Artwork, Z: gekauft
# RULES: X must be of type Subject
#        Y must be of type Verb
#        Z must be of type Object

# X: Person, Y: TicketFor, Z: bought
# X: Contact, Y: Account, Z: employedBy

option = {
    "template": {
        "x": "Contact",
        "y": "employedBy",
        "z": "Account"
    },
    "input": {
        "template": "x",
        "field": "contactFullName"
    },
    "queryTemplate": """
    SELECT ?accountName WHERE
    {            
        ?account a :Account ;
                 :accountName ?accountName .
        ?contact a :Contact ;
                 :contactFullName ?contactFullName ;
                 :employedBy ?account .            
        FILTER(?contactFullName = "$")
    }
    """
}

#"cnf": "q = A? * ∃C: Contact(C) ∧ Account(A?) ∧ employedBy(C, A?) ∧ contactFullName(C, '$')",
# Kunstwerk

#graph.query(
#    """
#    SELECT ?uni WHERE
#    {
#        TuringAward  win        ?person .
#        DeepLearning field      ?person .
#        ?person      university ?uni    .
#    }
#    """
#)

def build_prompt(prompt):
    return f"""
    Convert the following natrual language question into an conjunctive normal form:
    Question: {prompt}
    
    Just answer the CNF, nothing more, nothing less.
    Do not change the language, if you convert the user question into the CNF.
    For example: If the user writes in german, the answer should be in german!
    
    Take the following example in english as a guidance:
    Question: At what universities do the Turing Award winners in the field of Deep Learning work?
    Answer: q = U? * ∃V: win(TuringAward, V) ∧ field(DeepLearning, V) ∧ University(V, U?) 
    Bear in mind that the clauses here like win, field and University are only examples and can be different for the actual prompt. 
    """

def generate_cnf(query):
    return ollama.generate(
        model="llama3.2",
        prompt=build_prompt(query)
    )["response"]



