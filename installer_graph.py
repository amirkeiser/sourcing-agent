from typing import TypedDict, List, Annotated
from typing_extensions import NotRequired
from langchain_core.messages import AnyMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from search_node import SearchResponse, search_graph
from content_extraction_node import BusinessInformation, extract_business_info

# Initialize the LLM
model = ChatOpenAI(model="gpt-4o", temperature=0)

class LocationResponse(BaseModel):
    """Structured output for location extraction"""
    location: str = Field(description="The location/city name extracted from the query")

# Define our graph state
class InstallerState(MessagesState):
    """State for the installer discovery pipeline"""
    search_results: NotRequired[List[dict]]  # Stores search results
    extracted_info: NotRequired[List[dict]]  # Stores extracted business information
    current_location: NotRequired[str]  # Tracks the location being searched

def extract_location_node(state: InstallerState):
    """Node that extracts location from user query"""
    # Get the last message from the user
    last_message = state["messages"][-1]
    
    # Create input for location extraction
    location_input = {
        "messages": [
            ("system", """Extract the UK location from the user's query. 
            If no location is mentioned or the location is not in the UK, return an empty string.
            Examples:
            - "Find bathroom installers in Manchester" -> "Manchester"
            - "Get bathroom fitters near Leeds" -> "Leeds"
            - "Show me bathroom installers in New York" -> ""
            """),
            ("user", last_message.content if hasattr(last_message, 'content') else last_message['content'])
        ]
    }
    
    # Create a ReAct agent for location extraction
    location_graph = create_react_agent(
        model,
        tools=[],  # No tools needed for this task
        response_format=LocationResponse
    )
    
    # Extract location
    location_response = location_graph.invoke(location_input)
    location_info = location_response["structured_response"]
    location = location_info.location
    
    # Return appropriate response based on whether location was found
    if location:
        return {
            "current_location": location,
            "messages": [AIMessage(content=f"I'll search for bathroom installers in {location}. Please wait while I gather the information.")]
        }
    else:
        return {
            "current_location": "",
            "messages": [AIMessage(content="I couldn't identify a valid UK location in your request. Please specify a city or area in the UK.")]
        }

def search_node(state: InstallerState):
    """Node that searches for installers in a given location"""
    location = state.get("current_location", "")
    if not location:
        return {
            "search_results": [],
            "messages": [AIMessage(content="I couldn't proceed with the search as no valid location was provided.")]
        }
    
    # Run the search graph
    search_input = {
        "messages": [
            ("user", f"Find bathroom installers in {location}")
        ]
    }
    search_response = search_graph.invoke(search_input)
    
    # Get confirmed installers
    candidates = search_response["structured_response"].candidates
    confirmed_installers = [c.dict() for c in candidates if c.is_installer]
    
    # Construct message based on results
    if confirmed_installers:
        message = f"I found {len(confirmed_installers)} bathroom installer{'s' if len(confirmed_installers) == 1 else 's'} in {location}. I'll now gather detailed information about each business."
    else:
        message = f"I couldn't find any confirmed bathroom installers in {location}. You might want to try searching in nearby areas."
    
    # Update state with search results
    return {
        "search_results": confirmed_installers,
        "messages": [AIMessage(content=message)]
    }

def extraction_node(state: InstallerState):
    """Node that extracts detailed information from installer websites"""
    search_results = state.get("search_results", [])
    extracted_info = []
    
    # Process each search result
    for result in search_results:
        url = result.get("url")
        if url:
            business_info = extract_business_info(url)
            extracted_info.append(business_info)
    
    # Construct message based on extraction results
    if extracted_info:
        message = f"I've gathered detailed information about {len(extracted_info)} business{'es' if len(extracted_info) > 1 else ''}. "
        
        # Add a summary of what was found
        services_found = sum(1 for info in extracted_info if info.get('services_offered'))
        contacts_found = sum(1 for info in extracted_info if info.get('phone_numbers') or info.get('email_addresses'))
        
        message += f"Found contact information for {contacts_found} business{'es' if contacts_found > 1 else ''} "
        
        # Create CSV format
        csv_headers = ["Business Name", "Phone Numbers", "Email Addresses", "Physical Address", "Services Offered", "Years in Business", "Website URL", "Confidence Score"]
        csv_content = ",".join(csv_headers) + "\n"
        
        for info in extracted_info:
            csv_row = [
                f'"{info.get("business_name", "")}"',
                f'"{"; ".join(info.get("phone_numbers", []))}"',
                f'"{"; ".join(info.get("email_addresses", []))}"',
                f'"{info.get("physical_address", "")}"',
                f'"{"; ".join(info.get("services_offered", []))}"',
                f'"{info.get("years_in_business", "")}"',
                f'"{info.get("website_url", "")}"',
                f'"{info.get("confidence_score", "")}"'
            ]
            csv_content += ",".join(csv_row) + "\n"
        
        message += "**Here's the data in CSV format that you can copy and paste into Excel or Google Sheets:**\n\n"
    else:
        message = "I wasn't able to extract detailed information from any of the installer websites. This might be due to website accessibility issues."
    
    # Update state with extracted information
    return {
        "extracted_info": extracted_info,
        "messages": [AIMessage(content=message), AIMessage(content=csv_content)]
    }

# Create the graph
workflow = StateGraph(InstallerState)

# Add nodes
workflow.add_node("extract_location", extract_location_node)
workflow.add_node("search", search_node)
workflow.add_node("extract", extraction_node)

# Define edges
workflow.add_edge("extract_location", "search")
workflow.add_edge("search", "extract")

# Set entry point
workflow.set_entry_point("extract_location")

# Compile the graph
graph = workflow.compile()

# Example usage
if __name__ == "__main__":
    # Initialize state with a user query
    initial_state = {
        "messages": [{"role": "user", "content": "Find bathroom installers in Manchester"}]
    }
    
    # Run the graph
    for event in graph.stream(initial_state, stream_mode="updates", subgraphs=False):
        print(event)
    
    # # Print results
    # print("\nSearch Results:")
    # for installer in result.get("search_results", []):
    #     print(f"\n- {installer['business_name']}")
    #     print(f"  URL: {installer['url']}")
    
    # print("\nExtracted Information:")
    # for info in result.get("extracted_info", []):
    #     print(f"\n- {info['business_name']}")
    #     print(f"  Phone: {info['phone_numbers']}")
    #     print(f"  Email: {info['email_addresses']}")
    #     print(f"  Services: {info['services_offered']}") 