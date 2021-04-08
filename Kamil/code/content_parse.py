from bs4 import BeautifulSoup
import html

def parse_content(content):
    try:
        soup = BeautifulSoup(content, 'html.parser')
        paragraphs = soup.find_all("p", {"data-component": "paragraph"})

        full_text = "\n".join([paragraph.get_text() for paragraph in paragraphs])
        full_text = html.unescape(full_text)
        return full_text
    except:
        return None

# Użycie:
# articles["paragraph"] = articles["content"].apply(lambda x: parse_content(x))