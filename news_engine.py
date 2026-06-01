import os
import json
import feedparser
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

def fetch_latest_market_news(max_items=10):
    # ET Markets Feed
    url = "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
    feed = feedparser.parse(url)
    
    news_items = []
    if hasattr(feed, 'entries') and feed.entries:
        for entry in feed.entries[:max_items]:
            news_items.append({
                "title": entry.title,
                "link": entry.link,
                "description": entry.description if hasattr(entry, 'description') else "",
                "time": entry.published if hasattr(entry, 'published') else (entry.updated if hasattr(entry, 'updated') else "Recently")
            })
    return news_items

def fetch_and_summarize_news():
    news_items = fetch_latest_market_news(max_items=10)
    if not news_items:
        return []
        
    prompt = "You are an expert financial analyst. Below are recent Indian market news items.\n"
    prompt += "For each item, provide a very short 'gist' (3-6 words, like 'Company X Stake Divestment') and a 'summary' (1-2 sentences explaining the core impact).\n"
    prompt += "Output MUST be a valid JSON object with a single key 'news' containing an array of objects. Each object must have 'title', 'gist', 'summary', 'link', and 'time'.\n"
    prompt += "Use the exact provided links and time strings.\n\n"
    
    for i, item in enumerate(news_items):
        prompt += f"Item {i+1}:\nTitle: {item['title']}\nTime: {item['time']}\nDescription: {item['description']}\nLink: {item['link']}\n\n"
        
    try:
        response = _groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful financial API that outputs pure JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        data = json.loads(result_text)
        return data.get("news", [])
    except Exception as e:
        print(f"News fetch error: {e}")
        # Fallback if Groq fails
        return [{"title": n['title'], "gist": "Breaking News", "summary": n['description'][:150] + "...", "link": n['link'], "time": n['time']} for n in news_items]

if __name__ == "__main__":
    news = fetch_and_summarize_news()
    print(json.dumps(news, indent=2))
