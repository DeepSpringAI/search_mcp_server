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
    """
    Get the full path of headers above the current element.
    Returns headers in order from highest level to lowest, separated by ' - '.
    
    Args:
        element: BeautifulSoup element
        
    Returns:
        str: Path of headers separated by ' - '
    """
    headers = []
    current = element
    last_level = 100
    print('\n')
    # Find all headers before the current element
    while current:
        # Find the previous header
        header = current.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if not header:
            break
        
        
        current_level = int(header.name[1])
        if current_level < last_level:
            # print(f"Level: {header.name[1]}, Text: {header.get_text().strip()}")
            # Add header text to the list
            headers.append(header.get_text().strip())
            last_level = current_level
        # Move to the header element to find its parent headers
        current = header
    
    # Reverse to get highest level first
    # headers.reverse()
    
    # Join headers with separator
    return ' - '.join(headers) if headers else ""

def process_table(table_element) -> str:
    """
    Convert a markdown table into text-based information.
    
    Args:
        table_element: BeautifulSoup table element
        
    Returns:
        str: Text representation of the table
    """
    rows = []
    headers = []
    
    # Get headers
    header_row = table_element.find('tr')
    if header_row:
        headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
    
    # Process each row
    for row in table_element.find_all('tr')[1:]:  # Skip header row
        cells = row.find_all(['td', 'th'])
        row_data = {}
        
        for i, cell in enumerate(cells):
            header = headers[i] if i < len(headers) else f"Column {i+1}"
            cell_text = cell.get_text().strip()
            # Extract link if present
            link = cell.find('a')
            if link and link.get('href'):
                cell_text = f"{cell_text} ({link.get('href')})"
            row_data[header] = cell_text
        
        # Create text representation of the row
        row_text = ", ".join([f"{k}: {v}" for k, v in row_data.items()])
        rows.append(row_text)
    
    return "\n".join(rows)

def parse_markdown_table(text: str) -> tuple[bool, list[dict]]:
    """
    Parse a markdown table from text.
    
    Args:
        text (str): Text content that might contain a markdown table
        
    Returns:
        tuple[bool, list[dict]]: (is_table, list of row dictionaries)
    """
    lines = text.strip().split('\n')
    if len(lines) < 3:  # Need at least header, separator, and one row
        return False, []
        
    # Check if it's a table by looking for the separator line
    separator_line = lines[1]
    if not all(c in '|:-' for c in separator_line):
        return False, []
        
    # Extract headers
    headers = [h.strip() for h in lines[0].split('|')[1:-1]]
    
    # Process rows
    rows = []
    for line in lines[2:]:
        if not line.strip():
            continue
        cells = [c.strip() for c in line.split('|')[1:-1]]
        if len(cells) != len(headers):
            continue
            
        row_data = {}
        for i, cell in enumerate(cells):
            # Extract link if present
            link_match = re.search(r'\[([^\]]+)\]\(([^)]+)\)', cell)
            if link_match:
                text = link_match.group(1)
                url = link_match.group(2)
                cell = f"{text} ({url})"
            row_data[headers[i]] = cell
            
        rows.append(row_data)
    
    return True, rows

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
            
        # Delete existing file and create output directory
        if output_path:
            if os.path.exists(output_path):
                os.remove(output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Read markdown content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            return False, f"File is empty: {file_path}"
        
        # Convert markdown to HTML
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find user query from headers
        user_query = None
        headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        for header in headers:
            header_text = header.get_text().strip()
            if 'user' in header_text.lower():
                # Get the next paragraph after this header
                next_p = header.find_next('p')
                if next_p:
                    user_query = next_p.get_text().strip()
                    # Remove this paragraph as it's the user query
                    next_p.decompose()
                break
        
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
                
            # Check if this paragraph should be deleted
            if text.strip() == "```markdown":
                continue
                
            # Check if this is a markdown table
            is_table, table_rows = parse_markdown_table(text)
            if is_table:
                # Convert table rows to text
                table_text = []
                for row in table_rows:
                    row_text = ", ".join([f"{k}: {v}" for k, v in row.items()])
                    table_text.append(row_text)
                text = "\n".join(table_text)
                
                # Create metadata for table
                metadata = {
                    'filename': os.path.basename(file_path),
                    'section_header': get_section_header(element),
                    'links': [],  # Links are already included in the text
                    'is_list_item': False,
                    'is_table': True,
                    'user_query': user_query if user_query else ""
                }
            else:
                # Create metadata for regular text
                metadata = {
                    'filename': os.path.basename(file_path),
                    'section_header': get_section_header(element),
                    'links': links,
                    'is_list_item': element.name == 'li',
                    'is_table': False,
                    'user_query': user_query if user_query else ""
                }
            
            # Process chunks immediately
            text_chunks = chunk_text(text, chunk_size=100)
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
    readme_path = "/home/agent/workspace/parquet_mcp_server/temp/search_results_20250330_095329.md"
    output_path = "/home/agent/workspace/parquet_mcp_server/temp/search_results_20250330_095329.parquet"
    
    print(process_markdown_file(readme_path, output_path))
    
