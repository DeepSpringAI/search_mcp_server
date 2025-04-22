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
from parquet_mcp_server.client import perform_search_and_scrape_async
import asyncio
from parquet_mcp_server.src.supabase_db import SupabaseDB


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables from .env file
load_dotenv()

# Initialize SupabaseDB
db = SupabaseDB()

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
                logging.warning(f"Scraping failed with status: {scrape_status.status}")
            
            # Add a delay between requests to avoid rate limiting
            time.sleep(2)
            
        except Exception as e:
            logging.error(f"Error scraping {url}: {e}")
            raise Exception(f"Error scraping {url}: {e}")
    
    return scrape_results 


async def perform_search_and_scrape(search_queries: list[str], page_number: int = 1) -> tuple[bool, str]:
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


    return await find_similar_chunks(search_queries) 
    

async def summary_with_ollama(text: str, user_query: str) -> str:
    """
    Process text with the Ollama model in chunks and ensure the final result is under 4000 characters.
    
    Args:
        text (str): The complete text to process
        user_query (str): The user's query
    
    Returns:
        str: Final response from the model under 4000 characters
    """
    logging.info("Starting summary_with_ollama processing")
    
    async def process_chunk(chunk: str) -> str:
        """Process a single chunk with the Ollama model."""
        try:
            prompt_content = f"This is the user input query: {user_query}\nand this is the extracted information from the internet. Please summarize the results but mention all the information related to user query. Don't forget to add the sources links: \n{chunk}"
            chunk_response = await ollama_model.ainvoke([HumanMessage(content=prompt_content)])
            return chunk_response.content
        except Exception as e:
            logging.error(f"Error processing chunk: {str(e)}")
            return ""

    async def process_text_in_chunks(input_text: str) -> str:
        """Process text in chunks and combine results."""
        chunk_size = 3000
        chunks = [input_text[i:i + chunk_size] for i in range(0, len(input_text), chunk_size)]
        logging.info(f"Split text into {len(chunks)} chunks of size {chunk_size}")
        
        # Process all chunks concurrently
        chunk_tasks = [process_chunk(chunk) for chunk in chunks]
        chunk_responses = await asyncio.gather(*chunk_tasks)
        
        # Combine all responses
        combined_response = "\n\n\n------------------------------------------------ \n\n\n".join(chunk_responses)
        logging.info(f"Combined response length: {len(combined_response)}")
        
        return combined_response

    # First pass: process the original text
    first_pass_result = await process_text_in_chunks(text)
    
    # If the result is still too long, process it again
    if len(first_pass_result) > 4000:
        logging.info("First pass result too long, processing again")
        final_result = await process_text_in_chunks(first_pass_result)
    else:
        final_result = first_pass_result
    
    logging.info(f"Final result length: {len(final_result)}")
    return final_result

async def find_similar_chunks(queries: list[str]) -> tuple[bool, str]:
    """
    Get information from the results of a previous search.
    
    Args:
        queries (list[str]): List of search queries to merge.
    
    Returns:
        tuple[bool, str]: (success status, message with similar text chunks)
    """
    logging.info(f"Starting find_similar_chunks with queries: {queries}")
    similarity_threshold = 0.55
    json_path = './output.json'

    # Merge queries with 'and'
    merged_query = ' and '.join(queries)
    logging.info(f"Merged query: {merged_query}")

    # Load the JSON file containing embeddings
    try:
        logging.info(f"Loading JSON file from {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded JSON file with {len(data)} entries")
    except Exception as e:
        logging.error(f"Error loading JSON file: {str(e)}")
        return False, f"Error loading JSON file: {str(e)}"

    # Extract embeddings and texts
    texts = [item['text'] for item in data]
    links = [item.get('metadata', {}).get('url', '') for item in data]
    embeddings = np.array([item['embed'] for item in data])
    logging.info(f"Extracted {len(texts)} texts and {len(embeddings)} embeddings")

    # Get query embedding
    logging.info("Generating query embedding")
    query_embeddings = get_embedding([merged_query])
    if not query_embeddings:
        logging.error("Failed to generate query embedding")
        return False, "Failed to generate query embedding"
    logging.info("Successfully generated query embedding")

    query_embedding = np.array(query_embeddings[0])

    # Calculate cosine similarity
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    # Get indices of chunks with similarity above threshold
    high_similarity_indices = np.where(similarities > similarity_threshold)[0]
    logging.info(f"Found {len(high_similarity_indices)} chunks with similarity above threshold {similarity_threshold}")

    # Prepare the output
    output_texts = [f"{texts[i]}\nSource: {links[i]}" for i in high_similarity_indices]
    output_message = "\n\n--------------------\n\n".join(output_texts)
    logging.info(f"Prepared output message with {len(output_texts)} chunks")

    # Process with Ollama model
    logging.info("Starting Ollama model processing")
    final_response = await summary_with_ollama(output_message, merged_query)
    logging.info("Successfully completed Ollama model processing")

    # Add all links to the final response
    final_response = f"{final_response}\n\n--------------------\n\nAll of the searched websites is listed here: \n - {'\n - '.join(list(set(links)))}"

    # Create tmp directory if it doesn't exist
    os.makedirs('./tmp', exist_ok=True)
    with open(f'./tmp/output_{int(time.time())}.txt', 'w', encoding='utf-8') as f:
        f.write(final_response)

    return True, final_response


if __name__ == "__main__":
    logging.info("Starting main execution")
    queries = ["آیفون ۱۶ قیمت"]
    logging.info(f"Running with queries: {queries}")
    success, message = asyncio.run(find_similar_chunks(queries))
    logging.info(message)