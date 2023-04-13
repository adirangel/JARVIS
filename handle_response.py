import re
import requests
from bs4 import BeautifulSoup
import webbrowser

def get_webpage_title(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.title.string.strip()

def open_url(url):
    webbrowser.open(url)

def handle_response(response):
    # Check if the response contains a URL
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    url_match = re.search(url_regex, response)

    if url_match:
        url = url_match.group(0)
        open_url(url)
        # Get the title of the webpage
        title = get_webpage_title(url)
        # Replace the URL in the response with the webpage title
        response = re.sub(url_regex, title, response)

    return response
