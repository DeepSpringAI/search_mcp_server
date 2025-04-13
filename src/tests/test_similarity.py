import sys
import os
from parquet_mcp_server.client import find_similar_chunks

def test_find_similar_chunks():
    """Test the find similar chunks functionality"""
    try:
        # Example test data
        test_data = {
            "queries": ["macbook"]
        }
        
        # Call the find similar chunks function
        result = find_similar_chunks(
            queries=test_data["queries"]
        )
        
        
        return result  # Return the success status
        
    except Exception as e:
        print(f"Find Similar Chunks Test Error: {str(e)}")
        return False

if __name__ == "__main__":
    result = test_find_similar_chunks()
    print(result)
