import os
import re
import json
import requests
import anthropic
from dotenv import load_dotenv
from typing import Generator

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
BRIGHT_DATA_API_KEY = os.getenv("BRIGHT_DATA_API_KEY")
SERP_ZONE = os.getenv("SERP_ZONE", "serp_api2")
UNLOCKER_ZONE = os.getenv("UNLOCKER_ZONE", "web_unlocker1")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

BRIGHT_DATA_ENDPOINT = "https://api.brightdata.com/request"
BD_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {BRIGHT_DATA_API_KEY}",
}

TOOLS = [
    {
        "name": "search_web",
        "description": (
            "Search Google for information about a company. "
            "Use this to find funding details, founders, product descriptions, news, and more."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, e.g. 'Stripe startup founders funding history'",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "scrape_url",
        "description": (
            "Fetch the full content of a webpage. Use this to read a company's About page, "
            "Crunchbase profile, or any relevant URL found in search results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to fetch, e.g. 'https://stripe.com/about'",
                }
            },
            "required": ["url"],
        },
    },
]

SYSTEM_PROMPT = """You are a startup research agent. Given a company name, research it thoroughly using web search and page scraping, then produce a structured markdown report.

Your report must cover:
- Company overview (what they do, founding year, HQ)
- Founders and key team members
- Funding history (rounds, total raised, key investors)
- Business model and target market
- Recent news or developments
- Notable partnerships or customers

Be factual. Only include information found via tools. If something is not found, say so."""


def search_web(query: str) -> str:
    search_url = f"https://www.google.com/search?q={requests.utils.quote(query)}&hl=en&gl=us"
    payload = {"zone": SERP_ZONE, "url": search_url, "format": "json"}

    try:
        resp = requests.post(BRIGHT_DATA_ENDPOINT, headers=BD_HEADERS, json=payload, timeout=30)
        resp.raise_for_status()
        outer = resp.json()

        body = outer.get("body", outer)
        if isinstance(body, str):
            body = json.loads(body)

        organic = body.get("organic", [])
        if not organic:
            return f"No results found for: {query}"

        output = f"Search results for '{query}':\n\n"
        for i, item in enumerate(organic[:5], 1):
            output += f"{i}. {item.get('title', '')}\n   URL: {item.get('link', '')}\n   {item.get('description', '')}\n\n"
        return output

    except requests.exceptions.RequestException as e:
        return f"Search failed: {str(e)}"


def scrape_url(url: str) -> str:
    payload = {"zone": UNLOCKER_ZONE, "url": url, "format": "raw"}

    try:
        r = requests.post(BRIGHT_DATA_ENDPOINT, headers=BD_HEADERS, json=payload, timeout=45)
        r.raise_for_status()

        text = r.text
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:4000] if len(text) > 4000 else text

    except requests.exceptions.HTTPError as e:
        return f"Could not scrape {url} (HTTP {e.response.status_code}). Try a different URL or search instead."
    except requests.exceptions.RequestException as e:
        return f"Could not scrape {url}: {str(e)}"


def run_agent_stream(company_name: str) -> Generator[dict, None, None]:
    """
    Generator that yields progress events as the agent works.
    Each event is a dict with a 'type' key:
      - {"type": "thinking", "text": "..."}
      - {"type": "tool_call", "name": "search_web"|"scrape_url", "input": {...}}
      - {"type": "tool_result", "name": "...", "preview": "..."}
      - {"type": "report", "content": "..."}
      - {"type": "error", "message": "..."}
    """
    messages = [
        {"role": "user", "content": f"Research this startup and produce a full report: {company_name}"}
    ]

    try:
        while True:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            for block in response.content:
                if hasattr(block, "text") and block.text.strip():
                    yield {"type": "thinking", "text": block.text.strip()}

            if response.stop_reason == "end_turn":
                final = next(
                    (block.text for block in response.content if hasattr(block, "text")), ""
                )
                yield {"type": "report", "content": final}
                return

            if response.stop_reason != "tool_use":
                yield {"type": "error", "message": f"Unexpected stop: {response.stop_reason}"}
                return

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                yield {"type": "tool_call", "name": block.name, "input": block.input}

                if block.name == "search_web":
                    result = search_web(block.input["query"])
                elif block.name == "scrape_url":
                    result = scrape_url(block.input["url"])
                else:
                    result = f"Unknown tool: {block.name}"

                preview = result[:200].replace("\n", " ")
                yield {"type": "tool_result", "name": block.name, "preview": preview}

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

    except Exception as e:
        yield {"type": "error", "message": str(e)}


def run_agent(company_name: str) -> str:
    """CLI entry point — prints progress and returns the final report string."""
    print(f"\nResearching: {company_name}\n{'='*50}")
    report = ""
    for event in run_agent_stream(company_name):
        if event["type"] == "thinking":
            print(event["text"])
        elif event["type"] == "tool_call":
            print(f"\n[Tool: {event['name']}] {json.dumps(event['input'])}")
        elif event["type"] == "tool_result":
            print(f"[Result preview] {event['preview']}...")
        elif event["type"] == "report":
            report = event["content"]
        elif event["type"] == "error":
            print(f"[Error] {event['message']}")
    return report


if __name__ == "__main__":
    import sys
    company = sys.argv[1] if len(sys.argv) > 1 else input("Enter company name: ")
    result = run_agent(company)
    print("\n" + "="*50 + "\nFINAL REPORT\n" + "="*50)
    print(result)
