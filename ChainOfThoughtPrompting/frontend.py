import streamlit as st
import main
import pandas as pd

# Überprüfe Verfügbarkeit der KI-Modelle
is_gemini_available = main.is_gemini_available()
is_llama_available = main.is_llama_available()
ai_model_options = []

# Erstelle Liste der verfügbaren KI-Modelle
if is_llama_available:
    ai_model_options.append("llama3.2")

if is_gemini_available:
    ai_model_options.append("Gemini 1.5 Pro")

# Haupttitel und Modellauswahl
st.title("Chain-of-Thought Prompting")

# Bereich für die Modellauswahl
st.subheader("Modellauswahl")
ai_model = None
if len(ai_model_options) != 0:
    ai_model = st.radio(label="KI-Modell", options=ai_model_options)

# Fehlermeldungen für nicht verfügbare Modelle
if len(ai_model_options) == 0:
    st.error("""
    **Es ist kein KI-Modell verfügbar.**  
    Folge den Anweisungen, um fortzufahren.
    """)

# Warnungen für fehlende Modelle
if not is_gemini_available:
    st.warning("""
    `Gemini 1.5 Pro` ist nicht verfügbar.  
    Erstelle im `ChainOfThoughtPrompting`-Ordner eine `.env`-Datei mit einem API-Schlüssel (Key-Name: `GENAI_API_KEY`)
    """)

if not is_llama_available:
    st.warning("""
    `llama3.1` ist nicht verfügbar.  
    Installiere ollama und führe dann `ollama run llama3.2` aus, um `llama3.2` zu installieren.
    """)

# Chain-of-Thought Modusauswahl
st.subheader("Einstellungen")
cot_mode = st.radio(label="Chain-of-Thought Modus", options=["Few-Shot", "Zero-Shot"])

st.divider()

# MatheBot Hauptbereich
st.header("MatheBot")
option = st.radio(label="Funktion", options=["Freies Prompting", "Benchmark"])

# Bereich für freies Prompting
if option == "Freies Prompting":
    # Informationstext und Beispiel
    st.info(
        """
        Frage den MatheBot eine einfache Matheaufgabe **auf Englisch** in natürlicher Sprache.  
        Zum Beispiel:  
        `A software company hires 15 developers in January and 10 more in February. If 5 developers resign in March, how many developers remain?`
        """
    )

    # Eingabebereich für den Benutzer
    user_prompt = st.text_area(label="Prompt")

    # Chain-of-Thought Optionen
    use_cot = st.checkbox(label="Nutze Chain-of-Thought", value=True)
    generate_additional_non_cot_answer = st.checkbox(
        label="Generiere zusätzliche Antwort, die kein Chain-of-Thought verwendet",
        disabled=not use_cot
    )

    # Button zur Anfrageverarbeitung
    fired = st.button("Frage stellen", disabled=len(ai_model_options) == 0)

    # Verarbeitung der Anfrage
    if fired:
        with st.spinner("Generiere Antwort(en)"):
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

# Bereich für Benchmark-Tests
elif option == "Benchmark":
    # Konfiguration der Benchmark-Parameter
    limit_benchmark_questions = st.checkbox(
        label="Limitiere maximale Anzahl an Benchmark-Prompts",
        value=False
    )
    max_benchmark_questions_input = st.number_input(
        label="Maximale Anzahl an Benchmark-Prompts",
        value=5,
        disabled=not limit_benchmark_questions,
        max_value=len(main.benchmark_questions),
        min_value=1
    )

    # Button zum Starten der Benchmark-Tests
    benchmark_fired = st.button("Benchmark starten", disabled=len(ai_model_options) == 0)

    # Ausführung und Anzeige der Benchmark-Ergebnisse
    if benchmark_fired:
        with st.spinner("Benchmark wird ausgeführt"):
            max_benchmark_questions = None
            if limit_benchmark_questions:
                max_benchmark_questions = max_benchmark_questions_input

            analysis = main.run_benchmark(ai_model, max_benchmark_questions)
            main.save_benchmark_analysis(analysis)

        st.success(
            f"""
            **Benchmark-Ergebnisse**  
            Anzahl der Benchmarks: `{analysis["amountOfBenchmarks"]}`  
            Richtige Nicht-CoT Antworten: `{analysis["amountOfCorrectNonCotAnswers"]}`  
            Richtige CoT Antworten: `{analysis["amountOfCorrectCotAnswers"]}`  
            Erfolgsrate der Nicht-CoT Antworten: `{int(analysis["successRateNonCotAnswers"] * 100)}%`  
            Erfolgsrate der CoT Antworten: `{int(analysis["successRateCotAnswers"] * 100)}%`  
            """
        )
        df = pd.DataFrame(analysis["results"])
        st.table(df)


















