import json
import sys
import os
from parquet_mcp_server.client import embed_parquet

def test_embed_parquet():
    """Test the embed-parquet functionality"""
    try:
        # Example test data
        test_data = {
            "input_path": "/home/agent/workspace/parquet_mcp_server/input.parquet",
            "output_path": "/home/agent/workspace/parquet_mcp_server/output.parquet",
            "column_name": "text",
            "embedding_column": "embeddings",
            "batch_size": 2
        }
        
        # Call the embedding function
        result = embed_parquet(
            input_path=test_data["input_path"],
            output_path=test_data["output_path"],
            column_name=test_data["column_name"],
            embedding_column=test_data["embedding_column"],
            batch_size=test_data["batch_size"]
        )
        
        print("Embedding Test Result:", result)
        return True
        
    except Exception as e:
        print(f"Embedding Test Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_embed_parquet()
    sys.exit(0 if success else 1) 