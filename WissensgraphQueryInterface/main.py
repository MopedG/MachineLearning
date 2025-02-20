
# WICHTIGER HINWEIS
# Damit der Subgraph angezeigt werden kann, ist zwingend die Installation der
# Software Graphviz (https://graphviz.org/) nötig!


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

# Erstelle RDF Graphen aus art_show_ontology.ttl Datei
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

"""
Gibt ein Bild des Subgraphen einer Abfragenvorlage in Bytes zurück
"""
def subgraph(query_template):
    # Hole Bedingungen, die für die übergebene Vorlage definiert wurden
    x_condition = query_template["subgraph"]["xCondition"] if "xCondition" in query_template["subgraph"] else ""
    z_condition = query_template["subgraph"]["zCondition"] if "zCondition" in query_template["subgraph"] else ""
    condition = query_template["subgraph"]["condition"] if "condition" in query_template["subgraph"] else ""


    # Erstelle SPARQL-Abfrage, die die Knoten des Subgraphen zurück gibt
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

    # Führe SPARQL-Abfrage aus
    subgraph_result = graph.query(subgraph_query)

    # Erstelle Pydot bzw. Graphviz Graphen zur Generierung der Abbildung
    pydot_graph = pydot.Dot(graph_type="graph")

    # Fülle Pydot bzw. Graphviz Graphen mit Knoten und Kanten basierend auf den SPARQL-Abfrage Ergebnissen.
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

    # Erstelle und Rückgabe der Abbildung des Subgraphen im 'GIF' Format als Bytes
    return pydot_graph.create(format="gif")

"""
Gibt die Resultate einer SPARQL-Abfrage zurück, die basierend auf der übergebenen
Vorlage, Abfragenvorlage und Benutzereingabe ausgeführt wird
"""
def query_ontology(query_template, query, user_input):
    query_result = graph.query(
        query["queryTemplate"].replace("$", user_input)
    )

    row_name = query_template["template"][query["output"]]["field"]

    result = []
    for row in query_result:
        result.append(f"{row[row_name]}")

    return result

"""
Baue KNF basierend auf Benutzereingabe und Abfragenvorlage
"""
def build_cnf(query_template, query_input, user_input):
    template = query_template["template"]

    # Hole Knotenbezeichnungen
    node_x = template["x"]["node"]
    node_y = template["y"]["node"]
    node_z = template["z"]["node"]

    # Bestimme Abkürzungen der Kontenbezeichnung
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

    # Bestimme die Knoten, die gesucht und gegeben sind
    if query_input == "x":
        exists_node_char = node_x_abbreviation
        searched_node_char = node_z_abbreviation
    else:
        exists_node_char = node_z_abbreviation
        searched_node_char = node_x_abbreviation

    # Baue KNF
    return f"q = {searched_node_char}? * ∃{exists_node_char}: {node_x}({node_x_abbreviation}{'?' if node_x_abbreviation == searched_node_char else ''}) ∧ {node_z}({node_z_abbreviation}{'?' if node_z_abbreviation == searched_node_char else ''}) ∧ {node_y}({node_x_abbreviation}{'?' if node_x_abbreviation == searched_node_char else ''}, {node_z_abbreviation}{'?' if node_z_abbreviation == searched_node_char else ''}) ∧ {field}({exists_node_char}, '{user_input}')"

# Abfragevorlagen dienen als Grundprinzip der App.
# Jede Vorlage definiert
# - ein Knoten-Kanten-Knoten Tripel.
# - Bedingungen, um den Subgraphen zu erstellen
# - Abfragen, die ausgeführt werden können
#       (z.B. Contact employedBy <Account?>, wobei der Benutzer einen Contact (contactFullName) eingibt und das System
#       den zugehörigen Account (accountName) sucht.
#       Pro Abfragevorlage können mehrere Abfragen definiert werden. Falls es mehrere Abfragen für eine Abfragevorlage
#       gibt, kann der Benutzer im Frontend entscheiden, welche der Abfragen er ausführen möchte,
#       in dem er das Eingabefeld leer lässt, welches er abfragen möchte.
#       Falls es nur eine Abfrage für eine Abfragevorlage gibt, wird im Frontend nur ein Eingabefeld angezeigt.

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
    },
    {
        "template": {
            "x": { "node": "Ticket", "field": "fairName" },
            "y": { "node": "boughtBy" },
            "z": { "node": "PersonAccount", "field": "fullName" }
        },
        "subgraph": {
            "xCondition": ":boughtBy ?z"
        },
        "queries": [
            {
                "input": "x",
                "output": "z",
                "example": "'ART SHOW YEAR 3' boughtBy ?",
                "queryTemplate": """
                SELECT ?fullName WHERE
                {            
                    ?personAccount a :PersonAccount ;
                             :fullName ?fullName .
                    ?ticket a :Ticket ;
                             :fairName ?fairName ;
                             :boughtBy ?personAccount .            
                    FILTER(?fairName = "$")
                }
                """
            },
            {
                "input": "z",
                "output": "x",
                "example": "? boughtBy 'Barack Obama'",
                "queryTemplate": """
                SELECT ?fairName WHERE
                {            
                    ?personAccount a :PersonAccount ;
                             :fullName ?fullName .
                    ?ticket a :Ticket ;
                             :fairName ?fairName ;
                             :boughtBy ?personAccount .            
                    FILTER(?fullName = "$")
                }
                """
            }
        ]
    },
    {
        "template": {
            "x": { "node": "Artwork", "field": "title" },
            "y": { "node": "artworkMedium" },
            "z": { "node": "ArtworkMedium", "field": "artMediumTitle" }
        },
        "subgraph": {
            "xCondition": ":artworkMedium ?z"
        },
        "queries": [
            {
                "input": "x",
                "output": "z",
                "example": "'Golden Statue' artworkMedium ?",
                "queryTemplate": """
                SELECT ?artMediumTitle WHERE
                {            
                    ?artworkMedium a :ArtworkMedium ;
                             :artMediumTitle ?artMediumTitle .
                    ?artwork a :Artwork ;
                             :title ?title ;
                             :artworkMedium ?artworkMedium .            
                    FILTER(?title = "$")
                }
                """
            },
            {
                "input": "z",
                "output": "x",
                "example": "? artworkMedium 'Painting'",
                "queryTemplate": """
                SELECT ?title WHERE
                {            
                    ?artworkMedium a :ArtworkMedium ;
                             :artMediumTitle ?artMediumTitle .
                    ?artwork a :Artwork ;
                             :title ?title ;
                             :artworkMedium ?artworkMedium .            
                    FILTER(?artMediumTitle = "$")
                }
                """
            }
        ]
    },
    {
        "template": {
            "x": { "node": "ArtworkSale", "field": "soldAtShow" },
            "y": { "node": "sellingAccount" },
            "z": { "node": "Account", "field": "accountName" }
        },
        "subgraph": {
            "xCondition": ":sellingAccount ?z"
        },
        "queries": [
            {
                "input": "x",
                "output": "z",
                "example": "'ART SHOW YEAR 3' sellingAccount ?",
                "queryTemplate": """
                SELECT ?accountName WHERE
                {            
                    ?account a :Account ;
                             :accountName ?accountName .
                    ?artworkSale a :ArtworkSale ;
                             :soldAtShow ?soldAtShow ;
                             :sellingAccount ?account .            
                    FILTER(?soldAtShow = "$")
                }
                """
            },
            {
                "input": "z",
                "output": "x",
                "example": "? sellingAccount 'Kunstmuseum AG'",
                "queryTemplate": """
                SELECT ?soldAtShow WHERE
                {            
                    ?account a :Account ;
                             :accountName ?accountName .
                    ?artworkSale a :ArtworkSale ;
                             :soldAtShow ?soldAtShow ;
                             :sellingAccount ?account .            
                    FILTER(?accountName = "$")
                }
                """
            }
        ]
    }
]


"""
Gibt die Knotenbezeichnungen der Abfragevorlagen zurück
"""
def get_template_node_sets():
    x = {query_template["template"]["x"]["node"] for query_template in query_templates}
    y = {query_template["template"]["y"]["node"] for query_template in query_templates}
    z = {query_template["template"]["z"]["node"] for query_template in query_templates}

    return x, y, z

"""
Query-Vorlage basierend auf Knotenbezeichnungen zurückgeben
"""
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

"""
Auswahl der Abfrage basierend auf Benutzereingabe
"""
def choose_query(query_template, user_input_x, user_input_z):
    if user_input_x is None or user_input_x.strip() == "":
        return user_input_z, _get_query(query_template, "z")

    if user_input_z is None or user_input_z.strip() == "":
        return user_input_x, _get_query(query_template, "x")


"""
Vorhersage des VIP-Status einer Person aus der Ontologie mit GraphSAGE und VIP-Bias
"""
def predict_vip_status_with_graphsage(graph, name):
    # Model und Daten vorbereiten
    model = SimpleGraphSAGE()
    data, name_to_idx, known_vips = prepare_graph_data(graph) # GraphSAGE-Daten vorbereiten
    
    # Wenn der VIP-Status (Bias) bereits bekannt ist, diesen direkt zurückgeben
    if name in known_vips:
        return known_vips[name]
    
    # Ansonsten GraphSAGE-Vorhersage machen
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # Modell trainieren
    model.train()
    
    # 200 Epochen trainieren und Cross-Entropy-Loss Funktion verwenden
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

"""
Trainingsdaten (VIP Status) aus der Ontologie extrahieren
"""
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


"""
Ontologie laden und RDF-Graph zurückgeben
"""
def load_rdf_graph(ontology_file = ontology_file):
    graph = Graph()
    graph.parse(ontology_file, format="ttl")
    return graph

"""
Alle Person-Accounts und Kontakte aus der Ontologie extrahieren
"""
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

"""
Initialisierung des GraphSAGE-Modells
"""
# Vereinfachte GraphSAGE-Implementierung
class SimpleGraphSAGE(torch.nn.Module):
    def __init__(self):
        super(SimpleGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels=3, out_channels=8)  # 3 Features: knows, relatedTo, bias
        self.conv2 = SAGEConv(in_channels=8, out_channels=2)  # 2 Ausgabeklassen

    # Forward-Pass
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

"""
Bereitet Daten für GraphSAGE vor mit zusätzlichem VIP-Bias
"""
def prepare_graph_data(graph):
    # GraphSAGE-Daten vorbereiten
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
    
    # Namen zu Index Mapping
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

    # Kantenliste in Tensor umwandeln
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y), name_to_idx, known_vips

"""
Findet ähnliche Kontakte basierend auf gemeinsamen Eigenschaften wie:
- Gleiche Kunstshow-Teilnahme
- Ähnliche Ticket-Typen
- Ähnlicher VIP-Status
- Gemeinsame Beziehungen
"""
def find_similar_contacts(graph, target_name):
    query = """
    SELECT DISTINCT ?otherName ?commonShow ?ticketType ?vipStatus WHERE {
        # Get target person/contact info
        {
            { ?target a :PersonAccount ; :fullName ?targetName }
            UNION
            { ?target a :Contact ; :contactFullName ?targetName }
        }
        
        # Get their ticket info
        ?targetTicket :boughtBy ?target ;
                     :fairName ?commonShow ;
                     :ticketType ?targetTicketType .
                     
        # Find others who attended same show
        {
            { ?other a :PersonAccount ; :fullName ?otherName }
            UNION
            { ?other a :Contact ; :contactFullName ?otherName }
        }
        ?otherTicket :boughtBy ?other ;
                     :fairName ?commonShow ;
                     :ticketType ?ticketType .
                     
        # Optional VIP status
        OPTIONAL {
            { ?other :VIPStatus ?vipStatus }
            UNION
            { ?target :VIPStatus ?vipStatus }
        }
        
        # Exclude the target person themselves
        FILTER(?targetName = "%s")
        FILTER(?otherName != ?targetName)
    }
    """
    
    results = graph.query(query % target_name)
    
    # Gruppierung der ähnlichen Kontakte
    similar_contacts = {}
    for row in results:
        name = str(row.otherName)
        if name not in similar_contacts:
            similar_contacts[name] = {
                'shows': set(),
                'ticketTypes': set(),
                'vipStatus': None
            }
        similar_contacts[name]['shows'].add(str(row.commonShow))
        similar_contacts[name]['ticketTypes'].add(str(row.ticketType))
        if row.vipStatus:
            similar_contacts[name]['vipStatus'] = str(row.vipStatus)
    
    return similar_contacts

"""
Gruppiert Kontakte und Person-Accounts basierend auf gemeinsamen Eigenschaften
"""
def get_marketing_groups(graph):
    query = """
    SELECT DISTINCT ?name ?show ?ticketType ?vipStatus WHERE {
        {
            { ?person a :PersonAccount ; :fullName ?name }
            UNION
            { ?person a :Contact ; :contactFullName ?name }
        }
        OPTIONAL {
            ?ticket :boughtBy ?person ;
                    :fairName ?show ;
                    :ticketType ?ticketType .
        }
        OPTIONAL {
            ?person :VIPStatus ?vipStatus
        }
    }
    """
    
    results = graph.query(query)
    
    groups = {
        'vip': set(),
        'premium': set(),
        'standard': set(),
        'shows': {}
    }
    
    # Gruppierung basierend auf gemeinsamen Eigenschaften
    for row in results:
        name = str(row.name)
        
        # Gruppierung nach VIP-Status
        if row.vipStatus and str(row.vipStatus) in ["VIP", "First Choice VIP"]:
            groups['vip'].add(name)
            
        # Gruppierung nach Ticket-Typ
        if row.ticketType:
            if str(row.ticketType) == "Premium-Ticket":
                groups['premium'].add(name)
            else:
                groups['standard'].add(name)
                
        # Gruppierung nach Kunstshow-Teilnahme
        if row.show:
            show = str(row.show)
            if show not in groups['shows']:
                groups['shows'][show] = set()
            groups['shows'][show].add(name)
    
    return groups
