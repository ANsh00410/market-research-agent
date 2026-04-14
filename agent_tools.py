# agent_tools.py

# Add these to the existing tools list in agent_tools.py

stock_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_analysis",
            "description": "Get technical stock analysis for Indian companies listed on NSE/BSE. Use this when the research topic involves publicly listed companies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "Company name e.g. 'Tata Motors', 'Reliance Industries'",
                    },
                    "ticker": {
                        "type": "string",
                        "description": "NSE ticker with .NS suffix e.g. 'TATAMOTORS.NS', 'RELIANCE.NS'. Optional.",
                    },
                },
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_sentiment",
            "description": "Analyze recent news sentiment for a company to understand if market mood is positive or negative.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "Company name e.g. 'Ola Electric', 'Zomato'",
                    }
                },
                "required": ["company_name"],
            },
        },
    },
]

# Merge with existing tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for market data, competitor info, trends, and statistics. Automatically searches in Indian context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query. Keep it specific. Example: 'electric vehicle market size revenue 2026'",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results. Default 6.",
                        "default": 6,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_indian_news",
            "description": "Fetch the latest Indian news articles on a topic. Use this to find recent developments, funding news, policy changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "News search query. Example: 'EV startups funding 2026'",
                    },
                    "max_articles": {
                        "type": "integer",
                        "description": "Number of articles. Default 5.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
]

tools = tools + stock_tools
