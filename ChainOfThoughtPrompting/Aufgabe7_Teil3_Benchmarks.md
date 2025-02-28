# Chain of Thought (CoT) Prompting - Performance Analyse

Dieses Projekt untersucht die Effektivität von Chain of Thought (CoT) Prompting im Vergleich zu Standard-Prompting bei mathematischen Fragestellungen.

## Methodik

Die Analyse basiert auf:
- 21 mathematischen Benchmark-Fragen basierend auf zuvor definierten Beispielfragen
- Vergleich zwischen CoT und Nicht-CoT Prompting
- Tests mit zwei KI-Modellen:
  - Gemini 1.5 Pro (Cloud-basiert)
  - Llama 3.2 (Lokal via Ollama)

## Benchmark-Ergebnisse

### Gemini 1.5 Pro
- Erfolgsrate mit CoT: 100%
- Erfolgsrate ohne CoT: 100%
- Volle Konsistenz bei beiden Ansätzen

### Llama 3.2
- Erfolgsrate mit CoT: 95.2%
- Erfolgsrate ohne CoT: 95.2%
- Gleichbleibende Performance bei beiden Ansätzen

## Schlussfolgerungen

1. Beide Modelle zeigen hohe Genauigkeit bei mathematischen Aufgaben
2. Überraschenderweise kein signifikanter Leistungsunterschied zwischen CoT und Standard-Prompting
3. Gemini 1.5 Pro zeigt leicht bessere Performance als Llama 3.2, jedoch nur marginal

### Mögliche Gründe für gleichbleibende Performance
- Evtl. waren die Benchmarkfragen zu simpel. Obwohl die für die Benchmarks genutzten Fragen sich an den Fragen aus dem Paper orientiert haben, kann es sein, dass die genutzen Modelle (Gemini 1.5 Pro & Llama3.2), die recht neu sind (im Vergleich zu den im Paper genutzen Modelle, welches 2022 erschienen ist) so gut / stark waren, dass sie selbst ohne Chain-of-Thought Prompting die Aufgaben lösen konnten.
- Evtl. ist Chain-of-Thought Promping bereits (teilweise oder nur in bestimmten Abfragen) in den genutzen Modellen implementiert.

## Benchmark Details

Die vollständigen Benchmark-Ergebnisse mit allen Antworten und Bewertungen sind im `benchmarks`-Ordner als JSON-Dateien gespeichert. Jede Datei enthält:
- Verwendetes KI-Modell
- Erfolgsraten
- Detaillierte Antworten für beide Prompting-Methoden
- Korrektheitsbewertung jeder Antwort
