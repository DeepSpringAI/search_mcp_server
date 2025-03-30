import os
import re
import json
import logging
import markdown
import pandas as pd
from bs4 import BeautifulSoup

def chunk_text(text: str, chunk_size: int = 100, overlap: float = 0.2) -> list[str]:
    """
    Split text into chunks of specified size with overlap.
    
    Args:
        text (str): Text to split
        chunk_size (int): Number of words per chunk
        overlap (float): Percentage of overlap between chunks
    
    Returns:
        list[str]: List of text chunks
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
        
    chunks = []
    overlap_size = int(chunk_size * overlap)
    start = 0
    
    while start < len(words):
        # Calculate end position for current chunk
        end = min(start + chunk_size, len(words))
        # Create chunk from words
        chunks.append(' '.join(words[start:end]))
        # Move start position, accounting for overlap
        start = end - overlap_size
        
    return chunks

def extract_text_and_links(element) -> tuple[str, list[str]]:
    """
    Extract text and links from a BeautifulSoup element.
    Preserves the original text structure and collects all links.
    """
    links = []
    text_parts = []
    
    for content in element.contents:
        if content.name == 'a':
            # It's a link
            href = content.get('href', '')
            if href:
                links.append(href)
            text_parts.append(content.get_text())
        else:
            # It's regular text
            text_parts.append(str(content))
    
    # Join all text parts and clean up
    full_text = ''.join(text_parts).strip()
    # Unescape HTML entities (like &amp;)
    full_text = BeautifulSoup(full_text, 'html.parser').get_text()
    
    return full_text, links

def get_section_header(element) -> str:
    """Get the nearest header above the current element."""
    header = element.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    return header.get_text() if header else ""

def process_markdown_file(file_path: str, output_path: str = None) -> tuple[bool, list[dict] | str]:
    """
    Process a markdown file into structured JSON and optionally save as parquet.
    
    Args:
        file_path (str): Path to the markdown file
        output_path (str, optional): Path to save the parquet file
        
    Returns:
        tuple[bool, list[dict] | str]: (success status, list of paragraph objects or error message)
    """
    try:
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
            
        # Create output directory if it doesn't exist
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
        # Read markdown content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            return False, f"File is empty: {file_path}"
        
        # Convert markdown to HTML
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        
        # Process paragraphs and list items
        chunks_data = []
        elements = soup.find_all(['p', 'li'])
        
        if not elements:
            return False, f"No processable content found in file: {file_path}"
            
        for element in elements:
            # Extract text and links
            text, links = extract_text_and_links(element)
            
            # Skip empty paragraphs
            if not text:
                continue
            
            # Get section header
            section = get_section_header(element)
            
            # Create metadata object
            metadata = {
                'filename': os.path.basename(file_path),
                'section_header': section,
                'links': links,
                'is_list_item': element.name == 'li'
            }
            
            # Process chunks immediately
            text_chunks = chunk_text(text, chunk_size=100)  # Increased chunk size to reduce number of chunks
            for chunk in text_chunks:
                chunks_data.append({
                    'text': chunk,
                    'metadata': json.dumps(metadata)
                })
                
                # Write to parquet in batches to save memory
                if len(chunks_data) >= 1000 and output_path:
                    try:
                        df = pd.DataFrame(chunks_data)
                        if os.path.exists(output_path):
                            df.to_parquet(output_path, index=False, append=True)
                        else:
                            df.to_parquet(output_path, index=False)
                        chunks_data = []  # Clear the buffer
                    except Exception as e:
                        return False, f"Error writing to parquet file: {str(e)}"
        
        # Write any remaining chunks
        if chunks_data and output_path:
            try:
                df = pd.DataFrame(chunks_data)
                if os.path.exists(output_path):
                    df.to_parquet(output_path, index=False, append=True)
                else:
                    df.to_parquet(output_path, index=False)
            except Exception as e:
                return False, f"Error writing final chunks to parquet file: {str(e)}"
            
        if not chunks_data:
            return False, f"No valid content chunks were generated from file: {file_path}"
            
        return True, f"Successfully saved {len(chunks_data)} markdown chunks to {output_path}"
        
    except Exception as e:
        error_msg = f"Error processing markdown file: {str(e)}"
        logging.error(error_msg)
        return False, error_msg 

if __name__ == "__main__":
    # Test processing of README.md
    readme_path = "/home/agent/workspace/parquet_mcp_server/src/parquet_mcp_server/README.md"
    output_path = "readme_processed.parquet"
    
    print(process_markdown_file(readme_path, output_path))
    
