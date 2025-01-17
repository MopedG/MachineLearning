from sqlite3.dbapi2 import paramstyle

from firecrawl import FirecrawlApp
from bs4 import BeautifulSoup

# FireCrawl-App initialisieren
app = FirecrawlApp(api_key="fc-f8ec1aa3e2e54d64acfefd7411b6b3aa")  # Ersetze YOUR_API_KEY durch deinen Schlüssel

# Liste der URLs
urls = [
    "https://www.artbasel.com/stories/suzanne-valadon-centre-pompidou-19th-century-women-artist",
    "https://www.artbasel.com/stories/seven-trailblazing-galleries-debuting-at-art-basel-hong-kong-in-2025",
    "https://www.artbasel.com/stories/paris-recommendations-david-lebovitz-gisela-mcdaniel-peter-freeman-justine-durrett-robbie-fitzpatrick",
    "https://www.artbasel.com/stories/notre-dame-de-paris-reopening-2025-secrets",
    "https://www.artbasel.com/stories/art-brut-outsider-art-art-basel-paris?lang=en",
    "https://www.artbasel.com/stories/christian-berst-art-brut-is-a-synthesis-between-the-personal-and-the-universal?lang=en",
    "https://www.artbasel.com/stories/art-basel-vip-collector-buy-art-in-person",
    "https://www.artbasel.com/stories/guide-art-basel-ubs-survey-of-global-collecting-2024",
    "https://www.artbasel.com/stories/art-collector-legacy-survey-legal-art-bequests",
    "https://www.artbasel.com/stories/american-collectors-art-market-2024",
    "https://www.artbasel.com/stories/great-wealth-transfer-survey-global-collecting-art-market-2024",
    "https://www.artbasel.com/stories/ieoh-ming-pei-life-in-architecture-retrospective-m-hong-kong",
    "https://www.artbasel.com/stories/architects-johnston-marklee-redefining-art-spaces-conversation-about-innovative-designs-collaborations",
    "https://www.artbasel.com/stories/simona-malvezzi-architecture-berlin",
    "https://www.artbasel.com/stories/havens-prisons-women-only-exhibitions",
    "https://www.artbasel.com/stories/women-collectors-patronage-lisa-perry-komal-shah-survey-global-collecting-ubs",
    "https://www.artbasel.com/stories/nova-sector-miami-beach-picks",
    "https://www.artbasel.com/stories/nft-crypto-art-market-transformation",
    "https://www.artbasel.com/stories/art-market-report-nfts-simon-denny",
    "https://www.artbasel.com/stories/digital-forum-nft-crypto-art",
    "https://www.artbasel.com/stories/art-basel-2022-exhibiting-metaverse",
    "https://www.artbasel.com/stories/kandis-williams-on-plants-and-the-black-body-night-gallery",
    "https://www.artbasel.com/stories/immersive-installations-digital-experiences-in-the-exhibition",
    "https://www.artbasel.com/stories/natasha-tontey-museum-macan-jakarta-audemars-piguet",
    "https://www.artbasel.com/stories/how-parley-for-the-oceans-is-fighting-to-save-our-seas",
    "https://www.artbasel.com/stories/artist-nolan-oswald-dennis-world-weather-network-lichen",
    "https://www.artbasel.com/stories/one-work-error-fabrice-hyber",
    "https://www.artbasel.com/stories/world-weather-network"
]

def debug_site(url):
    scrape_result = app.scrape_url(url, { 'formats': ['html'] })

    # Debugging: gesamten FireCrawl-Output ausgeben
    print(f"FireCrawl Output für {url}: {scrape_result}")

    html_content = scrape_result.get("html", "")
    if not html_content:
        print(f"Kein HTML-Inhalt für {url} gefunden.")
        return None

# Funktion zur Extraktion von Header, Subheader und Paragraphen
def extract_details(url):
    try:
        # Webseite mit FireCrawl scrapen
        scrape_result = app.scrape_url(url, { 'formats': ['html'] })
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
        print()

# debug_site("https://www.artbasel.com/stories/notre-dame-de-paris-reopening-2025-secrets")
