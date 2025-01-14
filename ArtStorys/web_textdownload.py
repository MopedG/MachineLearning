from firecrawl import FirecrawlApp
from bs4 import BeautifulSoup

# FireCrawl-App initialisieren
app = FirecrawlApp(api_key="fc-f8ec1aa3e2e54d64acfefd7411b6b3aa")  # Ersetze YOUR_API_KEY durch deinen Schlüssel

# Liste der URLs
urls = [
    "https://www.artbasel.com/stories/suzanne-valadon-centre-pompidou-19th-century-women-artist",
    "https://www.artbasel.com/stories/seven-trailblazing-galleries-debuting-at-art-basel-hong-kong-in-2025",
    "https://www.artbasel.com/stories/notre-dame-de-paris-reopening-2025-secrets"
]

# Funktion zur Extraktion von Header, Subheader und Paragraphen
def extract_details(url):
    try:
        # Webseite mit FireCrawl scrapen
        scrape_result = app.scrape_url(url)
        html_content = scrape_result.get("html", "")
        if not html_content:
            print(f"Kein Inhalt für {url} gefunden.")
            return None

        # HTML-Inhalt analysieren
        soup = BeautifulSoup(html_content, "html.parser")

        # Header extrahieren
        header = soup.find("h1").get_text(strip=True) if soup.find("h1") else "Keine Überschrift gefunden"

        # Subheader extrahieren
        subheader = soup.find("p", class_="subheading").get_text(strip=True) if soup.find("p", class_="subheading") else "Keine Unterüberschrift gefunden"

        # Paragraphen extrahieren
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if "subheading" not in p.get("class", [])]

        return header, subheader, paragraphs

    except Exception as e:
        print(f"Fehler beim Verarbeiten von {url}: {e}")
        return None

# Inhalte speichern
for url in urls:
    print(f"Verarbeite: {url}")
    result = extract_details(url)

    if result:
        header, subheader, paragraphs = result
        filename = f"{url.replace('https://', '').replace('http://', '').replace('/', '_')}.txt"
        with open(filename, "w", encoding="utf-8") as file:
            file.write(f"Header:\n{header}\n\n")
            file.write(f"Subheader:\n{subheader}\n\n")
            file.write("Content:\n" + "\n\n".join(paragraphs))
        print(f"Inhalte in '{filename}' gespeichert.")
    else:
        print(f"Keine Inhalte auf {url} gefunden oder Fehler aufgetreten.")
