from bing_search import get_bing_search_results
from generate_response import generate_response
from fetch_article import fetch_article

def search_and_analyze(query, bing_api_key, api_key, conversation_history):
    search_keywords = ["search for", "look for", "find me", "recent", "latest"]
    should_search = any(keyword in query.lower() for keyword in search_keywords)

    if should_search:
        search_results = get_bing_search_results(query, bing_api_key)
        if search_results:
            top_urls = [result["url"] for result in search_results[:5]]
            summaries = []

            # Define the max tokens for each article
            max_tokens_per_article = 800
            print("Visited websites:")
            for url in top_urls:
                print(url)
                article = fetch_article(url)

                # Truncate the article to max_tokens_per_article tokens
                tokens = article.split(" ")
                truncated_article = " ".join(tokens[:max_tokens_per_article])

                prompt = f"Read the following article and provide a brief summary or analysis:\n\n{truncated_article}"
                summary = generate_response(api_key, conversation_history, prompt)
                summaries.append(summary)
            summaries = "\n\n".join(summaries)
            final_prompt = f"Based on the following summaries, provide a single coherent answer to the question: '{query}'\n\n{summaries}"
            final_answer = generate_response(api_key, conversation_history, final_prompt)

            return search_results, final_answer
        else:
            return None, None
    else:
        return None, None
