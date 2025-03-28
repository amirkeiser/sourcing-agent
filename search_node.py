from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# Initialize Tavily client
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Initialize the LLM
model = ChatOpenAI(model="gpt-4o", temperature=0)

@tool
def search_installers(location: str, search_terms: Optional[List[str]] = None) -> str:
    """
    Search for bathroom installers in a specific location using Tavily.
    
    Args:
        location: The city or area to search in
        search_terms: Optional additional search terms
    """
    base_query = f"bathroom installers in {location}"
    if search_terms:
        base_query += " " + " ".join(search_terms)
    
    # Get search results with Tavily's specialized search
    search_result = tavily.search(
        query=base_query,
        search_depth="basic",
        max_results=10,  # Increased to get more potential matches
        include_raw_content=True
    )
    
    # Format results for LLM analysis
    formatted_results = []
    for result in search_result.get('results', []):
        formatted_results.append({
            "title": result.get("title", ""),
            "content": result.get("content", ""),
            "url": result.get("url", "")
        })
    
    return str(formatted_results)

class InstallerCandidate(BaseModel):
    """Simple format for validated bathroom installer businesses"""
    business_name: str = Field(description="Name of the business")
    url: str = Field(description="Website URL")
    is_installer: bool = Field(description="Whether this is confirmed to be a bathroom installation business")
    reason: str = Field(description="Brief reason why this was classified as installer or not")

class SearchResponse(BaseModel):
    """Simple response format for installer search"""
    candidates: List[InstallerCandidate] = Field(description="List of potential installer businesses found")

# Create the tools list
tools = [search_installers]

# System prompt for the agent
SYSTEM_PROMPT = """You are an expert at identifying legitimate bathroom installation businesses. Your task is to analyze search results and determine which ones are actual bathroom installers.

When evaluating each result, consider:
1. Does the business explicitly offer bathroom installation services?
2. Is it a direct service provider (not a directory or review site)?
3. Does it appear to be a legitimate business (not just a blog or news article)?

For each candidate:
- Set is_installer=True ONLY if you are confident it's a real bathroom installation business
- Provide a clear, concise reason for your decision
- Extract the actual business name (not the webpage title)
- Ensure the URL is the main business website

Exclude:
- Directory listings (e.g., Yelp, Yellow Pages)
- Review sites
- News articles
- Blog posts
- General contractors who don't specifically do bathrooms
- DIY guides or articles"""

# Create the graph
search_graph = create_react_agent(
    model,
    tools=tools,
    response_format=(SYSTEM_PROMPT, SearchResponse),
)

# Example usage
if __name__ == "__main__":
    inputs = {
        "messages": [
            ("user", "Find bathroom installers in Manchester")
        ]
    }
    response = search_graph.invoke(inputs)
    # Filter only confirmed installers
    confirmed_installers = [c for c in response["structured_response"].candidates if c.is_installer]
    print("\nConfirmed Bathroom Installers:")
    for installer in confirmed_installers:
        print(f"\n- {installer.business_name}")
        print(f"  URL: {installer.url}")
        print(f"  Reason: {installer.reason}") 