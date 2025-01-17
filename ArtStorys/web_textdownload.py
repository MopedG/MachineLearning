import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# List of URLs to scrape
urls = [
    "https://www.artbasel.com/stories/suzanne-valadon-centre-pompidou-19th-century-women-artist",
    "https://www.artbasel.com/stories/seven-trailblazing-galleries-debuting-at-art-basel-hong-kong-in-2025",
    "https://www.artbasel.com/stories/notre-dame-de-paris-reopening-2025-secrets"
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
