"""
agent.py — Groq-powered Indian Market Research Agent
Uses Llama 3.3 70B with tool-calling loop for deep market research.
"""

import os
import json
import time
from groq import Groq
from dotenv import load_dotenv

from tools import search_web, search_indian_news
from stock_tools import get_stock_analysis, get_stock_sentiment
from agent_tools import tools

load_dotenv()

# ─────────────────────────── Groq client ───────────────────────────────────

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

# ─────────────────────────── System prompt ─────────────────────────────────

SYSTEM_PROMPT = """You are an elite Indian Market Research Analyst with deep expertise in:

**Indian Market Context:**
- Indian consumer behavior across Tier 1 (Mumbai, Delhi, Bengaluru), Tier 2 (Pune, Hyderabad, Ahmedabad), and Tier 3 cities
- Indian regulatory landscape: SEBI, RBI, TRAI, FSSAI, MCA, Competition Commission of India (CCI)
- Government schemes: Make in India, Digital India, Startup India, PLI (Production Linked Incentive) schemes
- Indian startup ecosystem: Unicorns, Soonicorns, funding rounds (Pre-Seed to IPO), valuations in INR/USD
- Payment systems: UPI, BHIM, Paytm, PhonePe, NEFT/RTGS, RuPay
- Indian e-commerce, quick commerce, D2C brands, fintech, edtech, healthtech, agritech, EV, SaaS
- NSE/BSE stock market, Nifty 50, Sensex, FII/DII flows, SEBI regulations
- Indian demographics: 1.4B population, median age 28, rising middle class, rural-urban migration
- Pricing in INR, GST implications, import duties, MSME sector

**Your Research Methodology:**
1. ALWAYS conduct a minimum of 5-6 searches before writing the final report
2. Search for market size & TAM/SAM/SOM, key players, recent funding, trends, regulations
3. Cross-reference multiple sources for accuracy
4. Include specific INR/USD figures where available
5. Cite data points, statistics, and company examples

**Report Structure (always follow this):**
1. **Executive Summary** — 3-4 sentence overview with key market size
2. **Market Overview** — Size, growth rate (CAGR), TAM/SAM/SOM in INR
3. **Key Players & Competitive Landscape** — Top 5-8 companies with market share
4. **Current Trends & Innovations** — 4-6 major trends with examples
5. **Growth Opportunities** — Tier-wise breakdown, untapped segments
6. **Challenges & Risks** — Regulatory, competitive, macroeconomic risks
7. **Investment Landscape** — Recent funding rounds, VC interest, public market data
8. **Target Segments** — Demographics, psychographics, Tier 1/2/3 breakdown
9. **Strategic Recommendations** — 5 actionable recommendations for market entry/expansion

**Tone:** Professional, data-driven, actionable. Use ₹ symbol for INR values.
**Language:** English with Indian market terminology where appropriate.
"""

# ─────────────────────────── Tool map ──────────────────────────────────────

tool_map = {
    "search_web": search_web,
    "search_indian_news": search_indian_news,
    "get_stock_analysis": get_stock_analysis,
    "get_stock_sentiment": get_stock_sentiment,
}

# ─────────────────────────── Agent loop ────────────────────────────────────


def run_market_research_agent(
    topic: str, status_callback=None, progress_callback=None
) -> str:
    """
    Run the market research agent for a given topic.

    Args:
        topic:             The Indian market topic to research
        status_callback:   Optional callable(str) for live status updates
        progress_callback: Optional callable(float) for progress bar (0.0 – 1.0)

    Returns:
        Final market research report as a markdown string
    """

    def _log(msg: str):
        if status_callback:
            status_callback(msg)
        else:
            print(f"[Agent] {msg}")

    def _progress(val: float):
        if progress_callback:
            progress_callback(val)

    _log(f"Starting market research on: {topic}")
    _progress(0.05)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Conduct comprehensive market research on the Indian market for: **{topic}**\n\n"
                f"Requirements:\n"
                f"- Perform at least 5-6 distinct web/news searches\n"
                f"- Gather market size, key players, trends, funding, regulations\n"
                f"- Use actual data and statistics in INR where possible\n"
                f"- Write a professional, detailed report following the 9-section structure\n"
                f"- Be specific about Indian context (cities, regulations, payment methods, demographics)\n\n"
                f"Start researching now."
            ),
        },
    ]

    search_count = 0
    max_iterations = 20
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=4096,
                temperature=0.3,
            )
        except Exception as e:
            _log(f"Groq API error: {e}")
            time.sleep(2)
            continue

        choice = response.choices[0]
        message = choice.message

        # Append assistant message to history
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": (
                    [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in (message.tool_calls or [])
                    ]
                    if message.tool_calls
                    else None
                ),
            }
        )

        # ── Handle tool calls ─────────────────────────────────────────
        if message.tool_calls:
            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                _log(f"🔍 Calling tool: {fn_name}({_args_preview(fn_args)})")

                if fn_name in tool_map:
                    try:
                        result = tool_map[fn_name](**fn_args)
                        search_count += 1
                        _progress(min(0.1 + search_count * 0.1, 0.8))
                    except Exception as e:
                        result = {"error": str(e)}
                else:
                    result = {"error": f"Unknown tool: {fn_name}"}

                # Append tool result
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, default=str),
                    }
                )

        # ── Final answer ──────────────────────────────────────────────
        elif choice.finish_reason == "stop":
            if message.content and len(message.content) > 500:
                _log("✅ Research complete! Generating final report...")
                _progress(1.0)
                return message.content
            else:
                # Model stopped too early — push it to write the report
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"You've gathered enough data. Now write the complete, detailed market research "
                            f"report for '{topic}' following the 9-section structure. "
                            f"Include specific numbers, company names, and actionable insights. "
                            f"The report should be at least 1500 words."
                        ),
                    }
                )

        else:
            # Handle other finish reasons
            if message.content:
                _log(f"Finish reason: {choice.finish_reason}")
                if len(message.content) > 500:
                    _progress(1.0)
                    return message.content

    _log("⚠️ Max iterations reached. Returning best available report.")
    _progress(1.0)

    # Extract last substantial message
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and len(msg.get("content", "")) > 200:
            return msg["content"]

    return (
        f"# Market Research: {topic}\n\nUnable to complete research. Please try again."
    )


def _args_preview(args: dict) -> str:
    """Compact preview of tool arguments for logging."""
    preview = ", ".join(f"{k}={repr(v)[:40]}" for k, v in args.items())
    return preview[:100]


if __name__ == "__main__":
    # Quick test
    def status(msg):
        print(f"  STATUS: {msg}")

    report = run_market_research_agent("Quick Commerce market", status_callback=status)
    print("\n" + "=" * 60)
    print(report[:500])
    print("...")
