# agent.py
from groq import Groq
import json
import os
from dotenv import load_dotenv
from tools import search_web, search_indian_news
from agent_tools import tools
from stock_tools import get_stock_analysis, get_stock_sentiment

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are an expert Indian market research analyst with deep knowledge of:
- Indian economy, consumer behavior, and regional diversity
- Indian regulatory environment (SEBI, RBI, TRAI, etc.)
- Indian startup ecosystem (Tier 1, 2, 3 cities)
- Key Indian platforms: Flipkart, Meesho, Zepto, Ola, Paytm, etc.
- Indian demographics, income segments, and digital adoption trends

When researching a topic:
1. Search at least 5-6 times using different queries
2. Look for India-specific data: market size in INR/USD, Indian players, govt policies
3. Search for recent news on the topic
4. Consider Indian-specific factors: price sensitivity, UPI adoption, regional languages, rural vs urban

Your final report must include:
- 📊 Market Overview (size, growth rate, TAM in Indian context)
- 🏆 Key Indian Players & Competitors
- 📈 Market Trends (India-specific)
- 🌟 Opportunities & Market Gaps
- ⚠️ Challenges & Risks (regulatory, competition, infrastructure)
- 💰 Investment & Funding Landscape
- 🎯 Target Segments (metro, tier-2, rural, age groups)
- 📌 Conclusion & Recommendations

Always cite sources and mention if data is from 2025 or 2026."""

# Map tool names to actual functions
tool_map = {
    "search_web": search_web,
    "search_indian_news": search_indian_news,
    "get_stock_analysis": get_stock_analysis,
    "get_stock_sentiment": get_stock_sentiment,
}


def run_market_research_agent(topic: str) -> str:
    print(f"\n🇮🇳 Starting Indian Market Research on: {topic}\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Conduct comprehensive market research for the Indian market on: {topic}. Focus on India-specific data, players, trends, and opportunities.",
        },
    ]

    search_count = 0

    # Agent loop
    while True:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Free and powerful
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=4096,
            temperature=0.3,
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # If the model wants to use tools
        if finish_reason == "tool_calls" and message.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )

            # Execute each tool call
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                search_count += 1
                print(f"🔍 [{search_count}] {func_name}: {func_args.get('query', '')}")

                # Call the actual function
                result = tool_map[func_name](**func_args)

                # Add result to messages
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": result}
                )

        # Final answer
        elif finish_reason == "stop":
            final_report = message.content or ""
            print(f"\n✅ Research done! ({search_count} searches performed)\n")
            return final_report

        else:
            # Safety exit if something unexpected happens
            break

    return "Research could not be completed."
