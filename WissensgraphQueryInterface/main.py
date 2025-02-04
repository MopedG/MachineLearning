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
import pydot

"""
ERKENNTNISSE: NLP mit Spacy ist sehr unregelmäßig, d.h. es ist schwer, ein allgemeines zu finden, die Fälle dynamisch abdeckt.
- Es ist besser, die Entitäten und Schlüsselwörter entweder zum Teil dynamisch, zum Teil manuel zu extrahieren und dann die CNF-Formel zu generieren.
- Wir haben uns jedoch dazu entschlossen, einige Beispiel Queries zu definieren, die wir dann in CNF umwandeln.
"""

# import ollama
from owlready2 import get_ontology, sync_reasoner, default_world
from rdflib import Graph, Node, RDF

current_directory = os.path.dirname(os.path.abspath(__file__))
ontology_file = os.path.join(current_directory, 'art_show_ontology.ttl')

graph = Graph()
graph.parse(ontology_file, format="ttl")

# Welcher Account hat welches Artwork gekauft?
# Account gekauft Kunstwerk?
# X , Y , Z: X: Account, Y: Artwork, Z: gekauft
# RULES: X must be of type Subject
#        Y must be of type Verb
#        Z must be of type Object

# X: Person, Y: TicketFor, Z: bought
# X: Contact, Y: Account, Z: employedBy


def subgraph(query_template):
    x_condition = query_template["subgraph"]["xCondition"] if "xCondition" in query_template["subgraph"] else ""
    z_condition = query_template["subgraph"]["zCondition"] if "zCondition" in query_template["subgraph"] else ""
    condition = query_template["subgraph"]["condition"] if "condition" in query_template["subgraph"] else ""

    subgraph_query = f"""
    SELECT ?xField ?zField WHERE {{
        ?x a :{query_template["template"]["x"]["node"]} ;
                {f'{x_condition} ;' if x_condition else ""}
                :{query_template["template"]["x"]["field"]} ?xField .
    
        ?z a :{query_template["template"]["z"]["node"]} ;
                {f'{z_condition} ;' if z_condition else ""}
                :{query_template["template"]["z"]["field"]} ?zField .
                
        {condition}
    }}
    """

    subgraph_result = graph.query(subgraph_query)

    pydot_graph = pydot.Dot(graph_type="graph")

    edges = []
    x_nodes = {}
    z_nodes = {}
    for row in subgraph_result:
        x = str(row["xField"])
        z = str(row["zField"])

        edges.append((x, z))

        if not x in x_nodes:
            x_node = pydot.Node(name=x, label=x)
            x_nodes[x] = x_node
            pydot_graph.add_node(x_node)

        if not z in z_nodes:
            z_node = pydot.Node(name=z, label=z)
            z_nodes[z] = z_node
            pydot_graph.add_node(z_node)

    for x, z in edges:
        pydot_graph.add_edge(pydot.Edge(src=x_nodes[x], dst=z_nodes[z], label=query_template["template"]["y"]["node"]))

    return pydot_graph.create(format="gif")


def query_ontology(query_template, query, user_input):
    query_result = graph.query(
        query["queryTemplate"].replace("$", user_input)
    )

    row_name = query_template["template"][query["output"]]["field"]

    result = []
    for row in query_result:
        result.append(f"{row[row_name]}")

    return result


def build_cnf(query_template, query_input, user_input):
    template = query_template["template"]

    node_x = template["x"]["node"]
    node_y = template["y"]["node"]
    node_z = template["z"]["node"]

    node_x_abbreviation = ""
    node_z_abbreviation = ""
    while node_x_abbreviation == node_z_abbreviation:
        if len(node_x_abbreviation) == len(node_x) or len(node_z_abbreviation) == len(node_z):
            node_x_abbreviation = node_x[0] + "1"
            node_z_abbreviation = node_z[0] + "2"
            break

        node_x_abbreviation = node_x[:(len(node_x_abbreviation) + 1)]
        node_z_abbreviation = node_z[:(len(node_z_abbreviation) + 1)]

    field = template[query_input]["field"]

    if query_input == "x":
        exists_node_char = node_x_abbreviation
        searched_node_char = node_z_abbreviation
    else:
        exists_node_char = node_z_abbreviation
        searched_node_char = node_x_abbreviation

    return f"q = {searched_node_char}? * ∃{exists_node_char}: {node_x}({node_x_abbreviation}{'?' if node_x_abbreviation == searched_node_char else ''}) ∧ {node_z}({node_z_abbreviation}{'?' if node_z_abbreviation == searched_node_char else ''}) ∧ {node_y}({node_x_abbreviation}{'?' if node_x_abbreviation == searched_node_char else ''}, {node_z_abbreviation}{'?' if node_z_abbreviation == searched_node_char else ''}) ∧ {field}({exists_node_char}, '{user_input}')"


query_templates = [
    {
        "template": {
            "x": { "node": "Contact", "field": "contactFullName" },
            "y": { "node": "employedBy" },
            "z": { "node": "Account", "field": "accountName" }
        },
        "subgraph": {
            "xCondition": ":employedBy ?z"
        },
        "queries": [
            {
                "input": "x",
                "output": "z",
                "example": "'Henry Smith' employedBy ?",
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
            },
            {
                "input": "z",
                "output": "x",
                "example": "? employedBy 'Art Gallery Inc.'",
                "queryTemplate": """
                SELECT ?contactFullName WHERE
                {            
                    ?account a :Account ;
                             :accountName ?accountName .
                    ?contact a :Contact ;
                             :contactFullName ?contactFullName ;
                             :employedBy ?account .            
                    FILTER(?accountName = "$")
                }
                """
            }
        ]
    },
    {
        "template": {
            "x": { "node": "Account", "field": "accountName" },
            "y": { "node": "sold" },
            "z": { "node": "Artwork", "field": "title" }
        },
        "subgraph": {
            "condition": """
                ?artworkSale a :ArtworkSale ;
                             :soldArtwork ?z ;
                             :sellingAccount ?x .
            """
        },
        "queries": [
            {
                "input": "x",
                "output": "z",
                "example": "'Art Gallery Inc.' sold ?",
                "queryTemplate": """
                SELECT ?title WHERE
                {            
                    ?account a :Account ;
                             :accountName ?accountName .
                             
                    ?artwork a :Artwork ;
                             :title ?title .
                             
                    ?artworkSale a :ArtworkSale ;
                             :soldArtwork ?artwork ;
                             :sellingAccount ?account .
                            
                    FILTER(?accountName = "$")
                }
                """
            },
            {
                "input": "z",
                "output": "x",
                "example": "? sold 'The Great Painting'",
                "queryTemplate": """
                SELECT ?accountName WHERE
                {            
                    ?account a :Account ;
                             :accountName ?accountName .

                    ?artwork a :Artwork ;
                             :title ?title .

                    ?artworkSale a :ArtworkSale ;
                             :soldArtwork ?artwork ;
                             :sellingAccount ?account .

                    FILTER(?title = "$")
                }
                """
            }
        ]
    }
]


def get_template_node_sets():
    x = {query_template["template"]["x"]["node"] for query_template in query_templates}
    y = {query_template["template"]["y"]["node"] for query_template in query_templates}
    z = {query_template["template"]["z"]["node"] for query_template in query_templates}

    return x, y, z

def get_query_template(x, y, z):
    for query_template in query_templates:
        template = query_template["template"]
        if template["x"]["node"] == x and template["y"]["node"] == y and template["z"]["node"] == z:
            return query_template
    return None

def does_query_template_support_input(query_template, template):
    for query in query_template["queries"]:
        if query["input"] == template:
            return True
    return False

def does_query_template_support_multiple_queries(query_template):
    return len(query_template["queries"]) > 1

def get_examples_of_query_template_queries(query_template):
    return [query["example"] for query in query_template["queries"]]

def _get_query(query_template, query_input):
    for query in query_template["queries"]:
        if query["input"] == query_input:
            return query

def choose_query(query_template, user_input_x, user_input_z):
    if user_input_x is None or user_input_x.strip() == "":
        return user_input_z, _get_query(query_template, "z")

    if user_input_z is None or user_input_z.strip() == "":
        return user_input_x, _get_query(query_template, "x")







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



