import streamlit as st

import main

def input_validation(user_input_x, user_input_z):
    if user_input_x is None:
        if user_input_z.strip() == "":
            return "Es ist eine Eingabe erforderlich."

    if user_input_z is None:
        if user_input_x.strip() == "":
            return "Es ist eine Eingabe erforderlich."

    if user_input_z is not None and user_input_x is not None:
        if user_input_z.strip() == "" and user_input_x.strip() == "":
            return "Es ist eine Eingabe erforderlich."

        if (not user_input_z.strip() == "") and (not user_input_x.strip() == ""):
            return "Es darf maximal ein Eingabefeld mit Text gefüllt sein."

    return None


st.title("Abfrage der Kunstshow-Ontologie")
st.subheader("Wissensgraph Query Interface")

x, y, z = st.columns(3)

nodes_x, nodes_y, nodes_z = main.get_template_node_sets()

selection_x = x.selectbox(label="x", label_visibility="hidden", options=list(nodes_x))
selection_y = y.selectbox(label="y", label_visibility="hidden", options=list(nodes_y))
selection_z = z.selectbox(label="z", label_visibility="hidden", options=list(nodes_z))

query_template = main.get_query_template(selection_x, selection_y, selection_z)

if query_template is None:
    st.error("Invalide Kombination")
else:
    st.divider()
    input_x, input_y, input_z = st.columns(3, vertical_alignment="bottom")

    node_x = query_template["template"]["x"]["node"]
    user_input_x = None
    if main.does_query_template_support_input(query_template, "x"):
        user_input_x = input_x.text_input(label=node_x, placeholder=query_template["template"]["x"]["field"])
    else:
        input_x.text(node_x)

    input_y.text(query_template["template"]["y"]["node"])

    node_z = query_template["template"]["z"]["node"]
    user_input_z = None
    if main.does_query_template_support_input(query_template, "z"):
        user_input_z = input_z.text_input(label=node_z, placeholder=query_template["template"]["z"]["field"])
    else:
        input_z.text(node_z)

    query_examples = main.get_examples_of_query_template_queries(query_template)

    if main.does_query_template_support_multiple_queries(query_template):
        query_examples_string = '  \n'.join(map(lambda ex: f"`{ex}`", query_examples))
        st.info(
            f"""
            *Diese Kombination erlaubt mehrere Abfragemöglichkeiten.*  
            Zum Beispiel:  
            {query_examples_string}  
            **Lasse das eine Eingabefeld leer, um den Wert des anderen abzufragen.**
            """
        )
    else:
        st.info(
            f"""
            Beispielabfrage:  
            `{query_template["queries"][0]["example"]}`
            """
        )

    pressed = st.button("Abfragen")
    if pressed:
        error_text = input_validation(user_input_x, user_input_z)

        if error_text:
            st.error(error_text)
        else:
            user_input, query = main.choose_query(query_template, user_input_x, user_input_z)
            query_result_string = '  \n'.join(map(lambda res: f"`{res}`", main.query_ontology(query_template, query, user_input)))
            subgraph_img = main.subgraph(query_template)

            st.success(
            f"""
            **Ergebnis der Abfrage**  
            {query_result_string if len(query_result_string) != 0 else "*Kein Ergebnis*"}    
            **Konjunktive Normalform**  
            `{main.build_cnf(query_template, query["input"], user_input)}`
            """
            )
            st.markdown("#### Subgraph")
            st.image(subgraph_img)
    ontology_file = "art_show_ontology.ttl"
    graph = main.load_rdf_graph(ontology_file)
    model = main.GraphSAGE(in_channels=10, out_channels=2, hidden_channels=16) # 10 input features, 2 output classes, 16 hidden units
    model.eval()
    st.divider()
    st.subheader("VIP-Status Vorhersage für Personen und Kontakte")
    selected = st.text_input("Geben Sie den Namen eines Kontakts oder einer Person ein, für die der VIP-Status vorhergesagt werden soll:")
    st.info(
        f"""
        *Verfügbar:*  \n 
        Kontakte: Henry Smith, Sarah Benz \n 
        Personen: Barack Obama, Heinz Schuster, Tom Miller
        """
    )
    if st.button("VIP Status vorhersagen"):
        if selected:
            node_features = main.get_node_features(graph, selected).unsqueeze(0)
            num_nodes = node_features.size(0)
            edge_index = main.get_edges(num_nodes)

            with main.torch.no_grad():
                prediction = model(node_features, edge_index)
                vip_prob = main.torch.exp(prediction)[0, 1].item()
            st.write(f"Vorhersage-Wahrscheinlichkeit für VIP-Status: {vip_prob:.2f}")
            st.write("(Schwellenwert für VIP-Status: > 0.5)")

            if vip_prob > 0.5:
                st.success("Dieser Kontakt wird als VIP eingestuft!")
            else:
                st.warning("Dieser Kontakt wird nicht als VIP eingestuft.")