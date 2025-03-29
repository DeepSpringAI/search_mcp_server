import json
import sys
import os
from parquet_mcp_server.client import get_parquet_info

def test_parquet_info():
    """Test the parquet-information functionality"""
    try:
        # Example test data
        test_data = {
            "file_path": "/home/agent/workspace/parquet_mcp_server/input.parquet"
        }
        
        # Call the parquet info function
        result = get_parquet_info(test_data["file_path"])
        
        print("Parquet Info Test Result:", result)
        return True
        
    except Exception as e:
        print(f"Parquet Info Test Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_parquet_info()
    sys.exit(0 if success else 1) 