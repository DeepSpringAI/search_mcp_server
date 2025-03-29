import json
import sys
import os
from parquet_mcp_server.client import convert_to_duckdb

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_duckdb_conversion():
    """Test the convert-to-duckdb functionality"""
    try:
        # Example test data
        test_data = {
            "parquet_path": "/home/agent/workspace/parquet_mcp_server/input.parquet",
            "output_dir": "/home/agent/workspace/parquet_mcp_server/db_output"
        }
        
        # Call the DuckDB conversion function
        result = convert_to_duckdb(
            parquet_path=test_data["parquet_path"],
            output_dir=test_data["output_dir"]
        )
        
        print("DuckDB Conversion Test Result:", result)
        return True
        
    except Exception as e:
        print(f"DuckDB Conversion Test Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_duckdb_conversion()
    sys.exit(0 if success else 1) 