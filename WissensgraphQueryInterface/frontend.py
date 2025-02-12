import os
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
    graph = main.load_rdf_graph()
    person_list = main.get_info_from_ontology(graph)
    st.divider()
    st.subheader("VIP-Status Vorhersage für Personen und Kontakte")
    selected = st.selectbox("Wählen Sie eine Person oder einen Kontakt aus, für den der VIP-Status vorhergesagt werden soll:", options=person_list)
    st.info(
        f"""
        *Verfügbar:*  \n 
        Kontakte: Henry Smith, Sarah Benz \n
        Personen: Barack Obama, Heinz Schuster, Tom Miller
        """
    )
    if st.button("VIP Status vorhersagen"):
        if selected:
            graph = main.load_rdf_graph()
            #is_vip = main.predict_vip_status(graph, selected)  # debug only
            is_vip_graphsage = main.predict_vip_status_with_graphsage(graph, selected)

            # Direkte Ontologie-Abfrage
            
            #if is_vip:
            #    st.success(f"{selected} ist als VIP eingestuft! (Aus der Ontologie)")
            #else:
            #    st.warning(f"{selected} ist nicht als VIP eingestuft. (Aus der Ontologie)")
        
            
            # GraphSAGE Vorhersage
            if is_vip_graphsage:
                st.success(f"{selected} wird als VIP eingestuft!")
            else:
                st.warning(f"{selected} wird nicht als VIP eingestuft.")

    st.divider()
    st.subheader("Zielgruppenorientierte Marketing-Kampagnen")
    
    tab1, tab2 = st.tabs(["Ähnliche Kontakte finden", "Marketing Gruppen"])
    
    with tab1:
        person_list = main.get_info_from_ontology(graph)
        selected_person = st.selectbox(
            "Wählen Sie eine Person, um ähnliche Kontakte zu finden:",
            options=person_list
        )
        
        if st.button("Ähnliche Kontakte suchen"):
            if selected_person:
                similar = main.find_similar_contacts(graph, selected_person)
                if similar:
                    for name, info in similar.items():
                        shows = ", ".join(info['shows'])
                        tickets = ", ".join(info['ticketTypes'])
                        vip = info['vipStatus'] if info['vipStatus'] else "Kein VIP Status"
                        
                        st.markdown(f"""
                        **{name}**
                        - Gemeinsame Shows: {shows}
                        - Ticket-Typen: {tickets}
                        - VIP Status: {vip}
                        """)
                else:
                    st.info("Keine ähnlichen Kontakte gefunden.")
                    
    with tab2:
        if st.button("Marketing Gruppen anzeigen"):
            groups = main.get_marketing_groups(graph)
            
            st.markdown("### VIP Gruppe")
            st.write(", ".join(groups['vip']) if groups['vip'] else "Keine VIPs")
            
            st.markdown("### Premium Ticket Gruppe")
            st.write(", ".join(groups['premium']) if groups['premium'] else "Keine Premium-Ticket Inhaber")
            
            st.markdown("### Standard Ticket Gruppe")
            st.write(", ".join(groups['standard']) if groups['standard'] else "Keine Standard-Ticket Inhaber")
            
            st.markdown("### Nach Kunstshows")
            for show, attendees in groups['shows'].items():
                st.markdown(f"**{show}**")
                st.write(", ".join(attendees))