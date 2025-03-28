from typing import Optional, Dict, List, Any
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
def scrape_website(url: str) -> str:
    """
    Scrape a website's content using Tavily's extract API.
    
    Args:
        url: The website URL to scrape
    """
    # Use Tavily's extract method to get website content
    extracted_content = tavily.extract(
        urls=url,
        summarize=False,  # We want raw content
        include_raw_content=True
    )
    
    # Format the extracted content
    if isinstance(extracted_content, dict):
        # Single URL response
        return str(extracted_content)
    elif isinstance(extracted_content, list) and len(extracted_content) > 0:
        # List response - take first result since we only passed one URL
        return str(extracted_content[0])
    else:
        return str({"error": "No content extracted"})

class BusinessInformation(BaseModel):
    """Structured format for extracted business information"""
    business_name: str = Field(description="Official business name")
    phone_numbers: list[str] = Field(description="List of contact phone numbers")
    email_addresses: list[str] = Field(description="List of contact email addresses")
    physical_address: Optional[str] = Field(description="Business physical address if available")
    services_offered: list[str] = Field(description="List of bathroom-related services offered")
    years_in_business: Optional[int] = Field(description="Number of years in business if stated")
    website_url: str = Field(description="Main website URL")
    confidence_score: float = Field(description="Confidence in extracted information (0-1)")

# System prompt for the content extraction agent
SYSTEM_PROMPT = """You are an expert at analyzing bathroom installer websites and extracting business information. Your task is to analyze the scraped content and extract structured business details.

Focus on finding:
1. Official business name
2. All contact methods (phone, email)
3. Physical location/address
4. Specific bathroom services offered
5. Years in business
6. Coverage area/regions served
7. Professional certifications
8. Specializations in bathroom work

Guidelines:
- Only include information you find in the content
- Mark fields as None if information isn't available
- Include all phone numbers and emails found
- For services, focus on bathroom-specific offerings
- Assign a confidence score (0-1) based on information completeness
- Be precise with extracted information - don't make assumptions

Remember: It's better to return None than to guess or make assumptions about missing information."""

# Create the tools list
tools = [scrape_website]

# Create the extraction agent
extraction_graph = create_react_agent(
    model,
    tools=tools,
    response_format=(SYSTEM_PROMPT, BusinessInformation),
)

def extract_business_info(url: str) -> Dict:
    """
    Process a single installer website and extract structured business information.
    
    Args:
        url: Website URL to analyze
        
    Returns:
        Dict: Structured business information
    """
    inputs = {
        "messages": [
            ("user", f"Extract business information from {url}")
        ]
    }
    
    response = extraction_graph.invoke(inputs)
    return response["structured_response"].dict()

# Example usage
if __name__ == "__main__":
    test_url = "https://www.manchesterbathroomfitters.co.uk/"
    business_info = extract_business_info(test_url)
    print("\nExtracted Business Information:")
    for key, value in business_info.items():
        print(f"{key}: {value}") 