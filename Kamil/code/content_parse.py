from bs4 import BeautifulSoup
import html
import numpy as np

def parse_content(content):
    if (content is not np.nan):
        soup = BeautifulSoup(content, 'html.parser')
        paragraphs = soup.find_all("p")

        full_text = "\n".join([paragraph.get_text() for paragraph in paragraphs])
        full_text = html.unescape(full_text)
        return full_text
    else:
        return np.nan

# Użycie:
# articles["paragraph"] = articles["content"].apply(lambda x: parse_content(x))