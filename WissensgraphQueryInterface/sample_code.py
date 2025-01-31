import streamlit as st
import owlready2
from owlready2 import get_ontology, sync_reasoner, default_world
import rdflib
import os

# ✅ Ensure required packages are installed
try:
    import streamlit as st
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "The 'streamlit' package is not installed. Please install it using 'pip install streamlit'.")

# ✅ Set Streamlit execution context
if not hasattr(st, "runtime"):
    raise RuntimeError("Streamlit must be run using 'streamlit run your_script.py'.")

# ✅ Simulate button click in debugging mode
DEBUG_MODE = False  # Change to False in production

if "simulate_button" not in st.session_state:
    st.session_state.simulate_button = DEBUG_MODE  # Set button to True in debug mode


#############################
# 1) Load Ontology and Run Reasoner
#############################
def load_ontology():
    """
    Loads the ontology, runs reasoning, and retrieves inferred knowledge.
    """
    file_path = r"C:\Users\yourpath\art_show_ontology.rdf"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File not found at {file_path}")

    print(f"🔍 Ontology is loading...")
    onto = get_ontology(file_path).load()
    print(f"✅ Ontology loaded successfully!")

    # Gather statistics
    classes_list = list(onto.classes())
    individuals_list = list(onto.individuals())
    object_properties_list = list(onto.object_properties())
    data_properties_list = list(onto.data_properties())

    # Print statistics
    print("Parsed ontology statistics:")
    print(f"Number of Classes: {len(classes_list)}")
    print(f"Classes: {classes_list}")
    print(f"Number of Individuals: {len(individuals_list)}")
    print(f"Individuals: {individuals_list}")
    print(f"Number of Object Properties: {len(object_properties_list)}")
    print(f"Object Properties: {object_properties_list}")
    print(f"Number of Data Properties: {len(data_properties_list)}")
    print(f"Data Properties: {data_properties_list}")

    # ✅ Run reasoner to infer new facts
    print("🔍 Reasoning is starting...")
    with onto:
        sync_reasoner()  # Uses HermiT reasoner
    print("✅ Reasoning completed!")

    # ✅ Plausibility check inferred instances.
    # Example: Visitors
    visitor_class = onto.search_one(iri="*Visitor")
    if visitor_class:
        visitors = list(visitor_class.instances())
        if visitors:
            print(f"✅ Visitors detected: {len(visitors)}")
            for v in visitors:
                print(f"   - {v.name}")
        else:
            print("❌ No Visitors detected! Reasoning may not be working correctly.")
    else:
        print("❌ Visitor class not found! Check ontology structure.")

    reasoned_graph = default_world.as_rdflib_graph()

    # ✅ Retrieve only inferred knowledge
    inferred_graph = rdflib.Graph()
    for s, p, o in default_world.as_rdflib_graph().triples((None, None, None)):
        if (s, p, o) not in onto.world.as_rdflib_graph():
            inferred_graph.add((s, p, o))

    # ✅ List all inferred triples
    if not inferred_graph:
        print("❌ No inferred triples found.")
    else:
        print("🔍 Inferred Triples:")
        for triple in inferred_graph:
            print(triple)

    return onto, reasoned_graph


# Load ontology and reason over it
onto, reasoned_graph = load_ontology()


#############################
# 2) Retrieve Subgraph & Answer
#############################
def query_llm(query: str, subgraph_ttl: str):
    generated_text = ""
    sample_prompt = (
        "You are a knowledge assistant answering questions.\n\n"
        f"Here is some relevant knowledge graph data:\n\n{subgraph_ttl}\n\n"
        f"Now answer this question: {query}"
    )
    return generated_text


#############################
# 3) Retrieve Subgraph & Answer
#############################
def retrieve_subgraph_and_answer(g: rdflib.Graph, query: str, is_hardcoded_graph_query=True):
    """
    Generalized method to handle "how many" and "list all" queries for different entities.
    """
    ns = rdflib.Namespace("http://example.org/schema#")
    g.bind("schema", ns)

    subgraph = rdflib.Graph()

    # Default answer if no relevant data is found
    final_answer = "No relevant data found."

    # Generate the subgraph
    if is_hardcoded_graph_query:
        # Option 1: Get relevant predicates, objects, and subjects and add them to the subgraph to answer the query
        is_count_task = "how many" in query.lower()
        is_list_task = "list all" in query.lower()

        class_value = None
        if "visitors" in query.lower():
            class_value = ns.Visitor
        elif "accounts" in query.lower():
            class_value = ns.Account
        # Add more entity mappings as needed

        event_name = None
        if "year 3" in query.lower():
            event_name = "ART SHOW YEAR 3"

        if class_value and event_name:
            entity_count = 0
            entity_list = []

            print(f"🔍 Checking for {class_value} entities...")
            for entity in g.subjects(rdflib.RDF.type, class_value):
                #        for entity in g.subjects(rdflib.RDF.type, ns.Visitor):
                print(f"🔎 Found entity: {entity}")

                if class_value == ns.Visitor:
                    for ticket in g.subjects(ns.boughtBy, entity):
                        barcode_scanned = list(g.objects(ticket, ns.barcodeIsScanned))
                        fairname = list(g.objects(ticket, ns.fairName))

                        if barcode_scanned and fairname:
                            fairname_str = str(fairname[0])  # Convert Literal to string for comparison
                            if barcode_scanned[0] == rdflib.Literal(True) and fairname_str == event_name:
                                print(f"✅ Confirmed {class_value}: {entity}")
                                subgraph.add((entity, rdflib.RDF.type, class_value))
                                for p, o in g.predicate_objects(ticket):
                                    subgraph.add((ticket, p, o))
                                entity_count += 1
                                entity_list.append(entity)
                                break  # Prevent duplicate counting
                else:
                    subgraph.add((entity, rdflib.RDF.type, class_value))
                    entity_count += 1
                    entity_list.append(entity)

            subgraph_ttl = subgraph.serialize(format="turtle")
            final_answer = f"{entity_count}" if is_count_task else entity_list

            return subgraph_ttl, final_answer

        return "", 0

    else:
        # Option 2: Use SPARQL query to retrieve the subgraph
        query = """
        TASK 1: YOUR SPARQL QUERY GOES HERE
        """
        results = g.query(query)

        # Convert results into a subgraph
        subgraph = rdflib.Graph()
        for row in results:
            subgraph.add(row)

        # Serialize the subgraph as Turtle format for better readability
        subgraph_ttl = subgraph.serialize(format="turtle")

        # TASK 2: Send the retrieved subgraph and user query to LLM of your choice to generate the answer
        final_answer = query_llm(query, subgraph_ttl)

        return subgraph_ttl, final_answer


#############################
# 4) Streamlit App UI
#############################
st.title("GraphRAG - Art Show data")

st.markdown("## Step 1: Query the Art Show data")
user_query = ""
# ✅ Query in natural language
st.markdown("### 1.0 Graph query: Hardcoded vs. LLM")
is_hardcoded_graph_query = st.radio("Chose between hardcoded and LLM generated graph query", ("Hardcoded", "LLM"))
if is_hardcoded_graph_query == "Hardcoded":
    is_hardcoded_graph_query = True
else:
    is_hardcoded_graph_query = False

st.markdown("### 1.1 Query in natural language")
user_query = st.text_input("Enter your query here:")

# ✅ Sample Queries
st.markdown("### 1.2 Sample queries as reference")
sample_query_count = "How many visitors attended ART SHOW YEAR 3?"
sample_query_list = "List all visitors of ART SHOW YEAR 3?"

if st.button(sample_query_count):
    user_query = sample_query_count
elif st.button(sample_query_list):
    user_query = sample_query_list
elif st.session_state.simulate_button:  # Simulate button click in debugging mode
    st.session_state.simulate_button = False  # Reset after one execution
    st.write("Button clicked! Running query...")
    user_query = sample_query_count

# ✅ Use reasoned RDFLib graph
subgraph_ttl, final_ans = retrieve_subgraph_and_answer(reasoned_graph, user_query, is_hardcoded_graph_query)

# Show subgraph
st.markdown("## Step 2: Visualize relevant subgraph")
if subgraph_ttl:
    st.text(subgraph_ttl)
else:
    st.text("No relevant subgraph found.")

# Show final answer
st.markdown("## Step 3: Provide final answer")
st.write(final_ans)

