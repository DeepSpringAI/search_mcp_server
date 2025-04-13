import json
import sys
import os
from parquet_mcp_server.client import perform_search_and_scrape

def test_search_and_scrape():
    """Test the search and scrape functionality"""
    try:
        # Example test data
        test_data = {
            "queries": ["macbook"],
        }
        
        # Call the search and scrape function
        result = perform_search_and_scrape(
            queries=test_data["queries"],
        )
        
        return result  # Return the success status
        
    except Exception as e:
        print(f"Search and Scrape Test Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_search_and_scrape()
    sys.exit(0 if success else 1) 