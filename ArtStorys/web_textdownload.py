import requests
from bs4 import BeautifulSoup

# Liste der URLs, deren Inhalte heruntergeladen werden sollen
urls = [
    "https://www.artbasel.com/stories/suzanne-valadon-centre-pompidou-19th-century-women-artist",
    "https://www.artbasel.com/stories/seven-trailblazing-galleries-debuting-at-art-basel-hong-kong-in-2025",
    "https://www.artbasel.com/stories/notre-dame-de-paris-reopening-2025-secrets"
]

# Funktion, um Heading, Subheading und <p>-Inhalte einer Webseite zu extrahieren
def extract_paragraphs(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        # HTTP-Request mit User-Agent
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Fehler werfen bei HTTP-Problemen

        # HTML-Inhalt analysieren
        soup = BeautifulSoup(response.text, "html.parser")

        # Versuch, Heading (<h1>) zu finden
        heading = soup.find("h1")
        heading_text = heading.get_text(strip=True) if heading else "Kein Heading gefunden"

        # Versuch, Subheading (<p> mit Klasse 'subheading') zu finden
        subheading = soup.find("p", class_="subheading")
        subheading_text = subheading.get_text(strip=True) if subheading else "Kein Subheading gefunden"

        # Alle <p>-Tags finden und den Text extrahieren
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]

        # Heading und Subheading zu den Paragraphen hinzufügen
        content = [f"H1: {heading_text}", f"Subheading: {subheading_text}"] + paragraphs

        # Debug: Ausgeben der gefundenen Werte
        print(f"URL: {url}\nHeading: {heading_text}\nSubheading: {subheading_text}\n")
        return content

    except Exception as e:
        print(f"Fehler beim Verarbeiten von {url}: {e}")
        return []

# Inhalte speichern
for url in urls:
    print(f"Verarbeite: {url}")
    content = extract_paragraphs(url)

    if content:
        # Dateiname für die Textdatei
        filename = f"./Textfiles/{url.replace('https://', '').replace('http://', '').replace('/', '_')}.txt"
        # Datei öffnen und Inhalte schreiben
        with open(filename, "w", encoding="utf-8") as file:
            file.write("\n\n".join(content))
        print(f"Inhalte in '{filename}' gespeichert.")
    else:
        print(f"Keine Inhalte auf {url} gefunden oder Fehler aufgetreten.")
