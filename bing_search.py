import requests

def get_bing_search_results(query, api_key):
    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query}
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        results = response.json()["webPages"]["value"]
        return results
    else:
        print(f"Error: {response.status_code}")
        return None
