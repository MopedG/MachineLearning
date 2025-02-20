import json

import streamlit as st
import main
import pandas as pd

st.title("Chain-of-Thought Prompting")

st.subheader("Modellauswahl")
ai_model = st.radio(label="KI-Modell", options=["Gemini 1.5 Pro", "llama3.2"])

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
        response = main.ask_mathbot(
            {
                "aiModel": ai_model,
                "userPrompt": user_prompt,
                "useCot": use_cot,
                "cotMode": cot_mode,
                "generateAdditionalNonCotResponse": generate_additional_non_cot_answer
            }
        )

        st.success(response["response"])

        if response["explicitNonCotResponse"]:
            st.info(
                f"""
                **Antwort ohne Chain-of-Thought**  
                {response["explicitNonCotResponse"]}
                """
            )



elif option == "Benchmark":
    limit_benchmark_questions = st.checkbox(label="Limitiere maximale Anzahl an Benchmark-Prompts", value=False)
    max_benchmark_questions_text = st.number_input(label="Maximale Anzahl an Benchmark-Prompts", value=5, disabled=not limit_benchmark_questions)

    benchmark_fired = st.button("Benchmark starten")

    if benchmark_fired:
        results = main.run_benchmark(ai_model, max_benchmark_questions_text)

        st.success(
            f"""
            **Benchmark-Ergebnisse**  
            Anzahl der Benchmarks: `{results["amountOfBenchmarks"]}`  
            Richtige Nicht-CoT Antworten: `{results["amountOfCorrectNonCotAnswers"]}`  
            Richtige CoT Antworten: `{results["amountOfCorrectCotAnswers"]}`  
            Erfolgsrate der Nicht-CoT Antworten: `{results["successRateNonCotAnswers"]}`  
            Erfolgsrate der CoT Antworten: `{results["successRateCotAnswers"]}`  
            """
        )
        df = pd.DataFrame(results["results"])
        st.table(df)


















