import requests
from bs4 import BeautifulSoup

# Liste der URLs, deren Inhalte heruntergeladen werden sollen
urls = [
    "https://www.artbasel.com/stories/suzanne-valadon-centre-pompidou-19th-century-women-artist",
    "https://www.artbasel.com/stories/seven-trailblazing-galleries-debuting-at-art-basel-hong-kong-in-2025",
    "https://www.artbasel.com/stories/notre-dame-de-paris-reopening-2025-secrets"
]


# Funktion, um die <p>-Inhalte einer Webseite zu extrahieren
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

        # Alle <p>-Tags finden und den Text extrahieren
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        return paragraphs
    except Exception as e:
        print(f"Fehler beim Verarbeiten von {url}: {e}")
        return []


# Inhalte speichern
for url in urls:
    print(f"Verarbeite: {url}")
    paragraphs = extract_paragraphs(url)

    if paragraphs:
        filename = f"{url.replace('https://', '').replace('http://', '').replace('/', '_')}.txt"
        with open(filename, "w", encoding="utf-8") as file:
            file.write("\n\n".join(paragraphs))
        print(f"Inhalte in '{filename}' gespeichert.")
    else:
        print(f"Keine Inhalte auf {url} gefunden oder Fehler aufgetreten.")
