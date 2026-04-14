# main.py
from agent import run_market_research_agent
from datetime import datetime


def save_report(topic: str, report: str):
    date = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"india_market_{topic[:25].replace(' ', '_')}_{date}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# 🇮🇳 Indian Market Research Report\n\n")
        f.write(f"**Topic:** {topic}\n")
        f.write(f"**Date:** {datetime.now().strftime('%B %d, %Y')}\n\n")
        f.write("---\n\n")
        f.write(report)

    print(f"📄 Report saved: {filename}")
    return filename


if __name__ == "__main__":
    print("🇮🇳 Indian Market Research Agent")
    print("=" * 40)
    print("Example topics:")
    print("  - Quick commerce / q-commerce market")
    print("  - EdTech market in India")
    print("  - UPI and fintech payments")
    print("  - EV two-wheelers market")
    print("  - D2C beauty and skincare brands")
    print("=" * 40)

    topic = input("\nEnter your research topic: ")

    report = run_market_research_agent(topic)

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    save_report(topic, report)
# ```

# ---

## Your Folder Structure
# ```
# india_market_agent/
# ├── .env
# ├── tools.py
# ├── agent_tools.py
# ├── agent.py
# └── main.py
