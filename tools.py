# tools.py
from duckduckgo_search import DDGS
from newsapi import NewsApiClient
import os
from dotenv import load_dotenv

load_dotenv()

news_client = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))


def search_web(query: str, max_results: int = 6) -> str:
    """Search the web using DuckDuckGo — completely free."""
    try:
        # Add "India" to every query automatically for Indian context
        india_query = f"{query} India"

        with DDGS() as ddgs:
            results = list(ddgs.text(india_query, max_results=max_results))

        if not results:
            return "No results found."

        formatted = []
        for r in results:
            formatted.append(
                f"Title: {r['title']}\n"
                f"Source: {r['href']}\n"
                f"Summary: {r['body']}\n"
            )

        return "\n---\n".join(formatted)

    except Exception as e:
        return f"Search error: {str(e)}"


def search_indian_news(query: str, max_articles: int = 5) -> str:
    """Fetch latest Indian news on a topic using NewsAPI."""
    try:
        response = news_client.get_everything(
            q=f"{query} India",
            language="en",
            sort_by="relevancy",
            page_size=max_articles,
        )

        articles = response.get("articles", [])
        if not articles:
            return "No news articles found."

        formatted = []
        for a in articles:
            formatted.append(
                f"Headline: {a['title']}\n"
                f"Source: {a['source']['name']}\n"
                f"Published: {a['publishedAt'][:10]}\n"
                f"Summary: {a['description']}\n"
                f"URL: {a['url']}\n"
            )

        return "\n---\n".join(formatted)

    except Exception as e:
        return f"News search error: {str(e)}"
