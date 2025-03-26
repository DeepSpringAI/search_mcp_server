import json
import sys
from parquet_mcp_server.client import embed_parquet

def main():
    if len(sys.argv) < 2:
        print("Please provide a JSON string as argument")
        print("Example: python test_mcp_embedding.py '{\"input_path\": \"sample.parquet\", \"output_path\": \"output.parquet\", \"column_name\": \"text\", \"embedding_column\": \"embeddings\"}'")
        sys.exit(1)

    try:
        # Parse the JSON argument
        args = json.loads(sys.argv[1])
        
        # Call the embedding function
        result = embed_parquet(
            input_path=args.get("input_path"),
            output_path=args.get("output_path"),
            column_name=args.get("column_name"),
            embedding_column=args.get("embedding_column")
        )
        
        print("Result:", result)
        
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 