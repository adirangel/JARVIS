# fetch_article.py
import requests
from bs4 import BeautifulSoup

def fetch_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    article_text = ' '.join([p.text for p in soup.find_all('p')])
    return article_text
