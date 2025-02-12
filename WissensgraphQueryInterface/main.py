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
import random

import pydot

"""
ERKENNTNISSE: NLP mit Spacy ist sehr unregelmäßig, d.h. es ist schwer, ein allgemeines zu finden, die Fälle dynamisch abdeckt.
- Es ist besser, die Entitäten und Schlüsselwörter entweder zum Teil dynamisch, zum Teil manuel zu extrahieren und dann die CNF-Formel zu generieren.
- Wir haben uns jedoch dazu entschlossen, einige Beispiel Queries zu definieren, die wir dann in CNF umwandeln.
"""


# Add Graphviz to PATH, necessary for pydot on WINDOWS
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
import pydot
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
# import ollama
from owlready2 import get_ontology
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

    # x condition -> meaning this condition exists on the x node

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



# Vereinfachte Version für VIP-Klassifizierung
def predict_vip_status(graph, name):
    """
    Überprüft den VIP-Status einer Person direkt aus der Ontologie
    """
    query = """
    SELECT ?name ?vipStatus WHERE {
        {
            ?personAccount a :PersonAccount ;
                           :fullName ?name ;
                           :VIPStatus ?vipStatus .
        }
        UNION
        {
            ?contact a :Contact ;
                     :contactFullName ?name ;
                     :VIPStatus ?vipStatus .
        }
    }
    """
    results = list(graph.query(query))
    
    for result in results:
        if str(result.name) == name:  # Direkter Vergleich des Namens
            vip_status = str(result.vipStatus)
            return vip_status in ["VIP", "First Choice VIP"]
    
    return False

def predict_vip_status_with_graphsage(graph, name):
    # Model und Daten vorbereiten
    """Vorhersage des VIP-Status mit GraphSAGE und Bias"""
    model = SimpleGraphSAGE()
    data, name_to_idx, known_vips = prepare_graph_data(graph) # GraphSAGE-Daten vorbereiten
    
    # Wenn der VIP-Status bereits bekannt ist, diesen direkt zurückgeben
    if name in known_vips:
        return known_vips[name]
    
    # Ansonsten GraphSAGE-Vorhersage machen
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # Modell trainieren
    model.train()
    
    # Mehr Epochen und bessere Verlustfunktion
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
    
    # Modell evaluieren und Vorhersage treffen
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred_probabilities = F.softmax(out, dim=1)
        pred = pred_probabilities.argmax(dim=1)
        
        if name in name_to_idx:
            idx = name_to_idx[name]
            return bool(pred[idx])
    
    return False

# Die folgenden Funktionen bleiben unverändert:
# - extract_training_data (für Informationszwecke)
# - load_rdf_graph
# - get_info_from_ontology


# Trainingsdaten (VIP Status) aus der Ontologie extrahieren -> WORKS!
def extract_training_data(graph):
    query = """
    SELECT ?name ?vipStatus ?knows ?relatedTo WHERE {
        {
            ?personAccount a :PersonAccount ;
                           :fullName ?name ;
                           :VIPStatus ?vipStatus .
            OPTIONAL { ?personAccount :knows ?knows . }
            OPTIONAL { ?personAccount :relatedTo ?relatedTo . }
        }
        UNION
        {
            ?contact a :Contact ;
                     :contactFullName ?name ;
                     :VIPStatus ?vipStatus .
            OPTIONAL { ?contact :knows ?knows . }
            OPTIONAL { ?contact :relatedTo ?relatedTo . }
        }
    }
    """
    result = graph.query(query)
    data = []
    for row in result:
        name = str(row.name)
        vip_status = 1 if str(row.vipStatus) in ["VIP", "First Choice VIP"] else 0
        knows = str(row.knows) if row.knows else None
        related_to = str(row.relatedTo) if row.relatedTo else None
        data.append((name, vip_status, knows, related_to))
    return data


# Ontologie laden
def load_rdf_graph(ontology_file = ontology_file):
    graph = Graph()
    graph.parse(ontology_file, format="ttl")
    return graph

# Alle Person-Accounts und Contacts aus der Ontologie extrahieren
def get_info_from_ontology(graph):
    query = """
    SELECT ?name WHERE {
        {
            ?personAccount a :PersonAccount ;
                           :fullName ?name .
        }
        UNION
        {
            ?contact a :Contact ;
                     :contactFullName ?name .
        }
        }
    """
    result = graph.query(query)
    return [str(row.name) for row in result]


# TODO: Graphsage trainiert model implizit, allerdings kann hier noch dazu die Cross entropy genutzt werden

# Grundidee: möglichst große klassifikation präzise (im Bezug auf Kreuzentropie) zu sagen ob der Node ein VIP ist oder nicht.

# AUFGABE b) Wie kann graphen erweitern für Zielgruppen orientierte Marketing campaigns?
# Ansatz "Clustering", wie kann ich die gemeinsamkeiten sammeln um die zusammen zu bewerten?
# Ansatz: Man könnte alle gemeinsamkeiten betrachten, weg von GraphSAGE, man hat die Knoten und evtl. einen fiktiven link "interesse", das verbindet
# die Knoten, die gemeinsamkeiten haben
# Empfehlung: schauen wieviele Nodes traversieren muss um links zu bilden zwischen den Nodes.
# Von einer Node alle relationen betrachten! Also: Knoten Obama, wie ist es bei traversal auf ticketkauf? Gibt es vielleicht eine weitere Node
# (Contact oder Person) die an der selben Kunstshow teilgenommen hat? Das ist eine Zielgruppe!
# Idee: unterschiedliche Kunstshow arten und Jahre!
# Einzelne Knoten mit bestimmter Tiefe (Layer) traversen und gucken ob dann bei dem Zielknoten eine Verbindung zu Nodes besteht
# Kein GraphSAGE notwendig, da wir nur die Embedings betrachten, sondern oberflächlich den Graphen anschauen und traversieren von Knoten zu knoten und
# schauen welche Ähnliche Eigenschaften aufweisen
# TODO: In Person Kontakte und Personen (Accounts) sind die Knoten, die müssen mit aktivierungsfunktion
# (rectified linear unit) durchführen und irgendwie trainieren mit regulation,
# danach kriegen wir representation im vektorraum (Von Wörter bzw. Token wie beim ersten mal zu Vektoren)

# Vereinfachte GraphSAGE-Implementierung
class SimpleGraphSAGE(torch.nn.Module):
    def __init__(self):
        super(SimpleGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels=3, out_channels=8)  # 3 Features: knows, relatedTo, bias
        self.conv2 = SAGEConv(in_channels=8, out_channels=2)  # 2 Ausgabeklassen

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# GraphSAGE-Daten vorbereiten
def prepare_graph_data(graph):
    """Bereitet Daten für GraphSAGE vor mit zusätzlichem VIP-Bias"""
    query = """
    SELECT ?name ?vipStatus ?knows ?relatedTo WHERE {
        {
            ?personAccount a :PersonAccount ;
                           :fullName ?name .
            OPTIONAL { ?personAccount :VIPStatus ?vipStatus }
            OPTIONAL { ?personAccount :knows ?knows }
            OPTIONAL { ?personAccount :relatedTo ?relatedTo }
        }
        UNION
        {
            ?contact a :Contact ;
                     :contactFullName ?name .
            OPTIONAL { ?contact :VIPStatus ?vipStatus }
            OPTIONAL { ?contact :knows ?knows }
            OPTIONAL { ?contact :relatedTo ?relatedTo }
        }
    }
    """
    results = list(graph.query(query))
    
    names = [str(row.name) for row in results]
    name_to_idx = {name: i for i, name in enumerate(names)}
    
    # Erstelle Feature-Matrix
    x = torch.zeros((len(names), 3))  # 3 Features: knows, relatedTo, vip_bias
    y = torch.zeros(len(names), dtype=torch.long)
    
    # Kanten (Beziehungen) als Kantenliste
    edge_index = [[], []]
    
    # VIP-Status aus der Ontologie als Bias hinzufügen
    known_vips = {}
    for row in results:
        name = str(row.name)
        if row.vipStatus:
            is_vip = str(row.vipStatus) in ["VIP", "First Choice VIP"]
            known_vips[name] = is_vip
    
    for i, row in enumerate(results):
        name = str(row.name)
        
        # VIP Bias setzen (Feature 3)
        if name in known_vips:
            x[i, 2] = 1.0 if known_vips[name] else -1.0
            y[i] = 1 if known_vips[name] else 0
        
        # Beziehungen verarbeiten
        if row.knows:
            known_name = str(row.knows)
            if known_name in name_to_idx:
                edge_index[0].append(i)
                edge_index[1].append(name_to_idx[known_name])
                x[i, 0] = 1  # knows Feature
                
        if row.relatedTo:
            related_name = str(row.relatedTo)
            if related_name in name_to_idx:
                edge_index[0].append(i)
                edge_index[1].append(name_to_idx[related_name])
                x[i, 1] = 1  # relatedTo Feature

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y), name_to_idx, known_vips
