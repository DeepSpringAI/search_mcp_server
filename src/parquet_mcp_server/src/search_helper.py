import json
import logging
import tempfile
from pathlib import Path
import re 
import requests
import os
from dotenv import load_dotenv
import os
import time
from firecrawl import FirecrawlApp
from datetime import datetime  # Import datetime module
import numpy as np
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import asyncio


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables from .env file
load_dotenv()


# Initialize Ollama LangChain model
ollama_model = ChatOllama(
    base_url=os.getenv("OLLAMA_URL"),
    model="llama3.1:8b",
)


def chunk_text(text: str, chunk_size: int = 500) -> list:
    """
    Split text into chunks of specified size.
    
    Args:
        text (str): Text to split
        chunk_size (int): Number of characters per chunk
    
    Returns:
        list: List of text chunks
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def get_embedding(texts: list) -> list:
    """
    Fetch embeddings for a batch of texts from the embedding server.

    Args:
        texts (list): A list of texts to generate embeddings for.
        url (str): The URL of the embedding server.

    Returns:
        list: A list of embeddings corresponding to the input texts.
    """
    model = "bge-m3"  # Changed from nomic-embed-text to bge-m3
    payload = {
        "model": model,
        "input": texts  # Pass all texts in a batch
    }
    logging.info('Make Embedding')
    try:
        response = requests.post(os.getenv('EMBEDDING_URL'), json=payload, verify=False)
        if response.status_code == 200:
            result = response.json()
            if 'embeddings' in result:
                return result['embeddings']
            else:
                logging.error(f"No embeddings found in response: {result}")
                return []
        else:
            logging.error(f"Error: {response.status_code}, {response.text}")
            return []
    except Exception as e:
        logging.error(f"Exception during request: {str(e)}")
        return []

def search_web(query, page=1):
    """
    Perform a web search using the SearchAPI.io API.
    
    Args:
        query (str): The search query
        page (int): The page number for pagination (default: 1)
        
    Returns:
        tuple: A tuple containing (organic_results, related_searches)
            - organic_results: List of search results
            - related_searches: List of related search queries
            
    Raises:
        ValueError: If API key is not found in environment variables
        requests.exceptions.RequestException: If there's an error making the HTTP request
        json.JSONDecodeError: If there's an error parsing the JSON response
    """
    url = "https://www.searchapi.io/api/v1/search"
    
    # Get API key from environment variables
    api_key = os.getenv("SEARCHAPI_API_KEY")
    if not api_key:
        logging.error("API key not found in environment variables")
        raise ValueError("API key not found. Please set SEARCHAPI_API_KEY in your .env file.")
    
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "page": page,
        "num": 3
    }
    
    try:
        logging.info(f"Making search request for query: {query}")
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        organic_results = data.get("organic_results", [])
        related_searches = data.get("related_searches", [])
        
        logging.info(f"Search successful for query: {query}")
        return organic_results, related_searches
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Error making search request: {str(e)}")
        raise RuntimeError(f"Request error: {str(e)}") from e
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON response: {str(e)}")
        raise ValueError(f"JSON parsing error: {str(e)}") from e

def get_links(markdown_content):
    """
    Filter markdown content by extracting all links and returning both the filtered content
    and a list of all links found.
    
    Args:
        markdown_content (str): The markdown content to filter
        
    Returns:
        tuple: A tuple containing (filtered_content, links)
            - filtered_content: The markdown content with all links removed
            - links: A list of all links found in the content
    """
    # Regular expression to match markdown links
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)|<([^>]+)>'
    
    # Find all links in the content
    link_matches = re.findall(link_pattern, markdown_content)
    
    # Extract the actual URLs from the matches
    links = []
    for match in link_matches:
        if isinstance(match, tuple):
            # For [text](url) format, the URL is the second element
            links.append(match[1])
        else:
            # For <url> format, the URL is the entire match
            links.append(match)
    
    
    # Clean up any double newlines that might have been created    
    return links 

def remove_markdown_links(text):
    # Remove markdown links while preserving the text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove any remaining URLs with percent encoding
    text = re.sub(r'https?://[^\s\]]+%[^\s\]]+', '', text)
    
    # Remove any remaining standalone URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Clean up any double newlines that might have been created
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text

def scrape_urls(organic_results):
    """
    Scrape each URL from the organic search results using Firecrawl API.
    
    Args:
        organic_results (list): List of organic search results
        
    Returns:
        dict: Dictionary mapping URLs to their scrape status and content
    """
    # Get Firecrawl API key from environment variables
    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
    if not firecrawl_api_key:
        raise ValueError("Firecrawl API key not found. Please set FIRECRAWL_API_KEY in your .env file.")
    
    # Initialize FirecrawlApp
    app = FirecrawlApp(api_key=firecrawl_api_key)
    
    # Dictionary to store scrape results for each URL
    scrape_results = {}
    
    # Scrape each URL
    for i, result in enumerate(organic_results):
        url = result.get("link")
        if not url:
            continue
        
        logging.info(f"Scraping URL {i+1}/{len(organic_results)}: {url}")
        
        try:
            # Scrape the URL
            scrape_status = app.scrape_url(
                url,
                params={'formats': ['markdown']}
            )

            # Store the scrape status and content if successful
            if scrape_status['metadata']['statusCode'] == 200:
                scrape_results[url] = {
                    'status': scrape_status['metadata']['statusCode'],
                    'content': scrape_status['markdown']
                }
                logging.info(f"Successfully scraped {url}")
            else:
                scrape_results[url] = {
                    'status': scrape_status['metadata']['statusCode'],
                    'error': f"Scraping failed with status: {scrape_status.status}"
                }
                logging.info(f"Scraping failed with status: {scrape_status.status}")
            
            # Add a delay between requests to avoid rate limiting
            time.sleep(2)
            
        except Exception as e:
            logging.error(f"Error scraping {url}: {e}")
            raise Exception(f"Error scraping {url}: {e}")
    
    return scrape_results 


def perform_search_and_scrape(search_queries: list[str], page_number: int = 1) -> tuple[bool, str]:
    """
    Perform searches and scrape URLs from the organic results for multiple queries.
    
    Args:
        search_queries (list[str]): The list of search queries to use.
        page_number (int): The page number for the search results.
    
    Returns:
        tuple[bool, str]: (success status, message)
    """
    all_results = []  # List to store all results with text and embeddings

    for search_query in search_queries:
        try:
            organic_results, related_searches = search_web(search_query, page_number)
        except Exception as e:
            return False, f"Error in SearchAPI: {str(e)}"
        
        # Log the search query results
        logging.info(f"Results for query '{search_query}' retrieved.")
        
        # Scrape URLs and save content as markdown files
        if organic_results:
            logging.info(f"Scraping URLs from organic search results for query '{search_query}'...")
            try:
                scrape_results = scrape_urls(organic_results)
            except Exception as e:
                return False, f"Error in Scraping {str(e)}"
        
            # Process and save markdown content for successful scrapes
            for i, (url, result) in enumerate(scrape_results.items()):
                if result['status'] == 200:
                    # Filter the markdown content
                    links = get_links(result['content'])
                    logging.info(f"Found {len(links)} links in {url}")

                    # Remove markdown links from the content
                    filtered_content = remove_markdown_links(result['content'])
                    
                    # Chunk the text
                    chunks = chunk_text(filtered_content, chunk_size=500)
                    
                    # Generate embeddings for all chunks
                    embeddings = get_embedding(chunks)  # Get embeddings for all chunks

                    # Combine text and embeddings into the result structure
                    current_date = datetime.now().strftime("%Y-%m-%d")  # Get current date as string
                    for chunk, embed in zip(chunks, embeddings):
                        all_results.append({
                            'text': chunk,
                            'metadata': {
                                'url': url,
                                'date': current_date  # Add current date to metadata
                            },
                            'embed': embed
                        })

    # Save all results to a JSON file in the current directory
    output_path = './output.json'
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(all_results, output_file, ensure_ascii=False, indent=4)
    
    logging.info(f"All results saved to {output_path}")


    return find_similar_chunks(search_queries) 
    

def process_chunks_with_ollama(chunks, user_query):
    """
    Process each chunk with the Ollama model and combine the results.
    
    Args:
        chunks (list): List of text chunks to process
        user_query (str): The user's query
    
    Returns:
        str: Combined response from the model
    """
    chunk_responses = []

    for chunk in chunks:
        prompt_content = f"This is the user input query: {user_query}\nand this is the extracted information from the internet. please extract all the information related to the user query based on these information: \n{chunk}"
        chunk_response = ollama_model.invoke([HumanMessage(content=prompt_content)])
        chunk_responses.append(chunk_response.content)

    combined_response = " ".join(chunk_responses)

    return combined_response

def find_similar_chunks(queries: list[str]) -> tuple[bool, str]:
    """
    Get information from the results of a previous search.
    
    Args:
        queries (list[str]): List of search queries to merge.
    
    Returns:
        tuple[bool, str]: (success status, message with similar text chunks)
    """
    similarity_threshold = 0.55
    json_path = './output.json'  # Always use this path

    # Merge queries with 'and'
    merged_query = ' and '.join(queries)

    # Load the JSON file containing embeddings
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON file: {str(e)}")
        return False, f"Error loading JSON file: {str(e)}"

    # Extract embeddings and texts
    texts = [item['text'] for item in data]
    embeddings = np.array([item['embed'] for item in data])

    # Get query embedding
    query_embeddings = get_embedding([merged_query])
    if not query_embeddings:
        logging.error("Failed to generate query embedding")
        return False, "Failed to generate query embedding"

    query_embedding = np.array(query_embeddings[0])

    # Calculate cosine similarity
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    # Get indices of chunks with similarity above threshold
    high_similarity_indices = np.where(similarities > similarity_threshold)[0]

    # Prepare the output
    output_texts = [texts[i] for i in high_similarity_indices]
    output_message = "\n--------------------\n".join(output_texts)

    # Process chunks with Ollama model
    chunk_size = 3000
    chunks = [output_message[i:i + chunk_size] for i in range(0, len(output_message), chunk_size)]
    final_response = process_chunks_with_ollama(chunks, merged_query)
    logging.info(f"final response =========== '{final_response}' retrieved.")
    file_path = "final_response_output.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(final_response)

    logging.info(f"Final response saved to {file_path}")

    

    return True, final_response

if __name__ == "__main__":
    # Example usage
    search_queries = ["آیفون ۱۶ قیمت"]  # Example queries
    success, message = perform_search_and_scrape(search_queries)
    logging.info(message)

    # queries = ["آیفون ۱۶ قیمت"]
    # success, message = find_similar_chunks(queries)
    # logging.info(message)