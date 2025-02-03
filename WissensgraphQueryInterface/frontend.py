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

selection_x = x.selectbox(label="x", options=list(nodes_x))
selection_y = y.selectbox(label="y", options=list(nodes_y))
selection_z = z.selectbox(label="z", options=list(nodes_z))

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
            **Lass das eine Eingabefeld leer, um den Wert des anderen abzufragen.**
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
            query_result_string = '  \n'.join(map(lambda res: f"`{res}`",main.query_ontology(query_template, query, user_input)))

            st.success(
            f"""
            **Ergebnis der Abfrage**  
            {query_result_string}    
            **Konjunktive Normalform**  
            `{main.build_cnf(query_template, query["input"], user_input)}`
            """
            )
