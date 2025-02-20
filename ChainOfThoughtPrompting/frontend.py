import streamlit as st
import main

st.title("Chain-of-Thought Prompting")

st.subheader("Modellauswahl")
st.radio(label="KI-Modell", options=["Gemini 1.5 Pro (>200B)", "llama3.1"])
st.warning("ACHTUNG: llama3.1 IST BAUSTELL!! NIX VERFÜGBAR DIESE MODELL")

st.subheader("Einstellungen")
cot_mode = st.radio(label="Chain-of-Thought Modus", options=["Few-Shot", "Zero-Shot"])

st.divider()

st.header("MatheBot")
option = st.radio(label="Funktion", options=["Freies Prompting", "Benchmark"])

if option == "Freies Prompting":
    st.info(
        """
        Frage den MatheBot eine einfache Matheaufgabe auf Englisch in natürlicher Sprache.  
        Zum Beispiel:  
        `A software company hires 15 developers in January and 10 more in February. If 5 developers resign in March, how many developers remain?`
        """
    )

    user_prompt = st.text_area(label="Prompt")

    use_cot = st.checkbox(label="Nutze Chain-of-Thought", value=True)
    generate_additional_non_cot_answer = st.checkbox(label="Generiere zusätzliche Antwort, die kein Chain-of-Thought verwendet", disabled=not use_cot)

    fired = st.button("Frage stellen")


    if fired:
        response = main.ask_mathbot(user_prompt, use_cot, cot_mode, generate_additional_non_cot_answer)

        st.success(response["response"])

        if response["explicitNonCotResponse"]:
            st.info(
                f"""
                **Antwort ohne Chain-of-Thought**  
                {response["explicitNonCotResponse"]}
                """
            )



elif option == "Benchmark":
    st.button("Benchmark starten")

st.divider()
















