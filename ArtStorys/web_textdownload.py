import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

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

# Initialize Selenium WebDriver
def init_driver():
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    driver.maximize_window()
    return driver

# Function to extract details from a webpage
def extract_details_with_selenium(driver, url):
    try:
        # Load the webpage
        driver.get(url)
        time.sleep(5)  # Wait for the page to load fully

        # Parse the page source with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Locate the content div
        content_div = soup.find('div', class_='content')
        if not content_div:
            print(f"Keine classe 'content' im div gefunden für {url}")
            return None

        # Extract main heading (h1)
        header = content_div.find("h1").get_text(strip=True) if content_div.find("h1") else "Keine Überschrift für den Artikel gefunden!"

        # Extract subheader (p with class 'subheading')
        subheader = content_div.find("p", class_="subheading").get_text(strip=True) if content_div.find("p", class_="subheading") else "Keine Unterüberschrift für den Artikel gefunden!"

        # Extract other paragraphs excluding subheader
        subheader_paragraphs = [
            p.get_text(strip=True) for p in content_div.find_all("p")
        ]

        paragraphs = [
            p.get_text(strip=True) for p in soup.find_all("p") if p not in subheader_paragraphs or p != subheader
        ]

        return header, subheader, subheader_paragraphs, paragraphs

    except Exception as e:
        print(f"Error while processing {url}: {e}")
        return None

# Save extracted content to a text file
def save_to_file(url, header, subheader, subheader_paragraphs, paragraphs):
    try:
        # Generate a safe filename from the URL
        filename = url.replace("https://", "").replace("http://", "").replace("/", "_") + ".txt"

        # Write content to the file
        with open(filename, "w", encoding="utf-8") as file:
            file.write(f"Überschrift:\n{header}\n\n")
            file.write(f"Unterüberschrift:\n{subheader}\n\n")
            file.write("Artikelinformation\n" +"\n\n".join(subheader_paragraphs))
            file.write("Inhalt:\n" + "\n\n".join(paragraphs))

        print(f"Inhalte in '{filename}' gespeichert.")

    except Exception as e:
        print(f"Kein HTML-Inhalt gefunden für {url} gefunden: Fehler {e}")

# Main script
def main():
    driver = init_driver()

    for url in urls:
        print(f"Verarbeite: {url}")
        result = extract_details_with_selenium(driver, url)

        if result:
            header, subheader, subheader_paragraphs, paragraphs = result
            save_to_file(url, header, subheader, subheader_paragraphs, paragraphs)
        else:
            print(f"Kein Inhalt für {url} gefunden")

    driver.quit()

if __name__ == "__main__":
    main()
