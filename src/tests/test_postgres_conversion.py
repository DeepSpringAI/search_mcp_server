import json
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parquet_mcp_server.client import convert_to_postgres

def test_postgres_conversion():
    """Test the convert-to-postgres functionality"""
    try:
        # Example test data
        test_data = {
            "parquet_path": "/home/agent/workspace/parquet_mcp_server/input.parquet",
            "table_name": "test_table"
        }
        
        # Call the PostgreSQL conversion function
        result = convert_to_postgres(
            parquet_path=test_data["parquet_path"],
            table_name=test_data["table_name"]
        )
        
        print("PostgreSQL Conversion Test Result:", result)
        return True
        
    except Exception as e:
        print(f"PostgreSQL Conversion Test Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_postgres_conversion()
    sys.exit(0 if success else 1) 