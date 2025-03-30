import json
import sys
import os
import pandas as pd
from parquet_mcp_server.client import chunk_markdown

def test_markdown_processing():
    """Test the markdown chunking functionality"""
    print("Starting markdown processing test...")
    # Use the project's README.md file
    readme_path = "/home/agent/workspace/parquet_mcp_server/src/parquet_mcp_server/README.md"
    output_path = "/home/agent/workspace/parquet_mcp_server/output/readme_chunks.parquet"
    
    print(f"Checking if README exists at: {readme_path}")
    if not os.path.exists(readme_path):
        print(f"Error: README.md not found at {readme_path}")
        return False
        
    print(f"README found. Creating output directory...")
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Calling chunk_markdown function...")
    # Call the markdown chunking function
    result = chunk_markdown(readme_path, output_path)
    
    print(result)
if __name__ == "__main__":
    success = test_markdown_processing()
    sys.exit(0 if success else 1) 