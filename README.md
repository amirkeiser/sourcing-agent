# Bathroom Installer Discovery Agent

This project implements a LangGraph agent designed to automate the discovery and basic information gathering of bathroom installers in specific UK locations.

## How it Works

The agent uses a multi-step process orchestrated by LangGraph:

1.  **`installer_graph.py`**: This is the main file that defines the LangGraph state and workflow.

    - It initializes the language model (LLM).
    - It defines the `InstallerState` which holds the data passed between steps (messages, search results, extracted info, current location).
    - It orchestrates the flow between the different nodes: `extract_location`, `search`, and `extract`.
    - It includes an example `if __name__ == "__main__":` block for testing the graph directly.

2.  **Location Extraction (`extract_location_node` in `installer_graph.py`)**:

    - Takes the user's initial query.
    - Uses a ReAct agent (with no tools) and a specific prompt to identify a UK location within the query.
    - Updates the state with the found `current_location` or provides feedback if no valid location is found.

3.  **Search (`search_node` in `installer_graph.py` and `search_node.py`)**:

    - Uses the `current_location` from the state.
    - Calls the `search_node.py` module, which defines:
      - A `search_installers` tool that uses the Tavily Search API to find businesses matching "bathroom installers in [location]".
      - A ReAct agent specifically prompted to analyze the search results and validate which ones are actual bathroom installation businesses (filtering out directories, blogs, etc.).
    - Updates the state with a list of `search_results` containing confirmed installer candidates (business name, URL).

4.  **Content Extraction (`extraction_node` in `installer_graph.py` and `content_extraction_node.py`)**:
    - Takes the list of URLs from the `search_results`.
    - For each URL, it calls the `content_extraction_node.py` module, which defines:
      - A `scrape_website` tool that uses Tavily's `extract` feature to get the raw content of the website.
      - A ReAct agent prompted to extract specific business information (name, phone, email, address, services, years in business) from the scraped content, structured according to the `BusinessInformation` Pydantic model.
      - Assigns a confidence score to the extracted information.
    - Updates the state with the `extracted_info`.
    - Formats the final extracted data into a CSV format within an AI message for easy copying.

## Setup and Running

1.  **Clone the Repository:**

    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create Environment File:**
    Create a file named `.env` in the root directory and add your API keys:

    ```.env
    TAVILY_API_KEY="your_tavily_api_key"
    OPENAI_API_KEY="your_openai_api_key"
    ```

3.  **Install Dependencies:**
    Make sure you have Python 3.8+ installed. Then, install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

    ```bash
    pip install -U "langgraph-cli[inmem]"
    ```

4.  **Run the LangGraph Development Server:**
    This command starts the LangGraph server, making your agent available. It specifies that the `graph` object within the `installer_graph` module should be exposed under the name `agent`.

    ```bash
    langgraph dev
    ```

    The server will typically run on `http://localhost:3000`.

5.  **Connect via Frontend:**

    - Go to [agentchat.vercel.app](https://agentchat.vercel.app/)
    - In the "Deployment URL" field, enter the address of your running LangGraph server (e.g., `http://localhost:3000`).
    - In the "Graph Name" field, enter `agent`.
    - Click "Connect".

6.  **Interact:**
    You can now send messages to the agent, for example:
    "Find bathroom installers in Birmingham"
    "Look for bathroom fitters near Bristol"

The agent will process your request through the defined steps and return the findings.
